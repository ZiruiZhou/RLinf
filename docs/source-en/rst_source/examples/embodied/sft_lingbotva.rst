LingBot-VA Supervised Fine-Tuning
==================================

This document explains how to run LingBot-VA supervised fine-tuning (SFT) in RLinf on the Libero-Object benchmark.

It targets the 5B-parameter LingBot-VA video-diffusion transformer on a single GPU with **≥45 GB** of memory (verified on NVIDIA L40), and reproduces the single-A100 reference recipe's loss curve while exceeding its ckpt_1000 evaluation success rate.

A more detailed end-to-end recipe lives at ``reproduce.md`` in the repository root.


Supported setups
----------------

Recommended config:

- ``examples/sft/config/libero_sft_lingbotva.yaml``: Libero-Object SFT, single-GPU recipe (bnb AdamW8bit + FSDP1 + flex_attention)

Starting point:

- **Cold-start from the LingBot-VA base checkpoint** (Wan-Video 2.2 backbone). Continuing from a partially trained checkpoint is supported but not the verified path.

Verified result on a single L40 (45 GB):

- ckpt_1000 mean SR over 10 tasks × 5 episodes: **26 % (13/50)** (reference single-A100 run: 18 %)
- Wall time to ckpt_1000: ~5 h (~17 s/step); to step 3000: ~14 h projected


Training entrypoint
-------------------

Use:

- ``examples/sft/run_vla_sft.sh``

The script runs:

.. code:: bash

   python examples/sft/train_vla_sft.py \
     --config-path examples/sft/config/ \
     --config-name libero_sft_lingbotva \
     runner.logger.log_path=<auto_log_dir>

Logs are written to:

- ``<repo>/logs/<timestamp>/run_embodiment.log``

The script also exports ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``, which is required for the 5B-param transformer plus activations to fit on a 45 GB card.


Environment
-----------

1. **PyTorch 2.9.0+cu126 is required.** Earlier versions hit a ``flex_attention`` + inductor codegen bug (``NameError: name 's10' is not defined``). Install with::

     pip install --no-deps --force-reinstall \
       torch==2.9.0+cu126 \
       --index-url https://download.pytorch.org/whl/cu126
     pip install nvidia-cufile-cu12 nvidia-nvshmem-cu12
     pip install --upgrade 'nvidia-nccl-cu12>=2.30.4'

2. Install RLinf and the LingBot-VA dependency group::

     pip install -e ${REPO_PATH}
     pip install -r ${REPO_PATH}/requirements/embodied/models/lingbotva.txt

3. Clone the lingbot-va peer repository (provides the ``wan_va`` Python package — needed at runtime by both training and the dataset-prep ``extract_latents.py``; the data-prep scripts themselves now live under ``toolkits/data_scripts_lingbotva/`` in this RLinf clone) and install in editable mode::

     git clone https://github.com/robbyant/lingbot-va.git <LINGBOT_VA_REPO_PATH>
     pip install -e <LINGBOT_VA_REPO_PATH>

4. Set the path env vars consumed by the launch script (substitute your own absolute paths)::

     export LINGBOT_VA_REPO_PATH=<your-lingbot-va-clone>
     export LINGBOT_VA_MODEL_PATH=<your-model-dir>
     export LINGBOT_VA_DATASET_PATH=<your-dataset-dir>
     export PYTHONPATH=${REPO_PATH}:${LINGBOT_VA_REPO_PATH}:${PYTHONPATH}
     export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

5. **lingbot-va flash_attn patch** (only needed on torch 2.9 without a custom flash_attn build). ``wan_va/modules/model.py`` imports ``flash_attn`` at module-load time. The prebuilt wheels do not load against the torch 2.9 ABI; patch the import to fall through gracefully:

   .. code:: diff

       try:
           from flash_attn_interface import flash_attn_func
      -except:
      -    from flash_attn import flash_attn_func
      +except Exception:
      +    try:
      +        from flash_attn import flash_attn_func
      +    except Exception:
      +        flash_attn_func = None

   This recipe uses ``attn_mode: flex`` and never calls ``flash_attn_func``, so the fallthrough is safe.


Data preparation
----------------

The training dataset is a LeRobot v2.1 conversion of 100 Libero-Object demonstrations (10 demos × 10 tasks) with pre-extracted Wan 2.2 VAE latents and a cached UMT5 empty-prompt embedding.

1. Download the raw HDF5 demonstrations. **Use** ``yifengzhu-hf/LIBERO-datasets``, **NOT** ``IPEC-COMMUNITY``: the two distributions have different wrist-camera mounts, and the IPEC version trains fine but evaluates at 0 % SR.

   .. code:: bash

      export LIBERO_RAW_DIR=<your-libero-raw-dir>
      huggingface-cli download yifengzhu-hf/LIBERO-datasets --local-dir ${LIBERO_RAW_DIR}

2. Run the three conversion scripts shipped under
   ``toolkits/data_scripts_lingbotva/``. See that directory's
   ``README.md`` for inputs / outputs of each step.

   .. code:: bash

      cd ${REPO_PATH}

      # (a) HDF5 -> LeRobot v2.1 format
      python toolkits/data_scripts_lingbotva/convert_libero_object_to_lerobot.py \
        --hdf5-root ${LIBERO_RAW_DIR}/libero_object \
        --output ${LINGBOT_VA_DATASET_PATH}_full

      # (b) Subsample to 10 demos x 10 tasks (deterministic via seed 42)
      python toolkits/data_scripts_lingbotva/select_subset.py \
        --src ${LINGBOT_VA_DATASET_PATH}_full \
        --dst ${LINGBOT_VA_DATASET_PATH} \
        --per-task 10 --seed 42

      # (c) Extract Wan 2.2 VAE latents + cache the UMT5 empty-prompt embedding
      python toolkits/data_scripts_lingbotva/extract_latents.py \
        --dataset ${LINGBOT_VA_DATASET_PATH} \
        --model-path ${LINGBOT_VA_MODEL_PATH}

3. Expected layout under ``${LINGBOT_VA_DATASET_PATH}``::

      meta/{info,episodes,episodes_stats,tasks}.jsonl
      data/chunk-000/episode_NNNNNN.parquet
      videos/chunk-000/observation.images.{agentview_rgb,eye_in_hand_rgb}/episode_NNNNNN.mp4
      latents/chunk-000/observation.images.{agentview_rgb,eye_in_hand_rgb}/episode_NNNNNN_<s>_<e>.pth
      empty_emb.pt

   If modern ``lerobot>=0.3.3`` rejects the dataset because ``episodes_stats.jsonl`` has image stats stored as ``(3,)`` instead of ``(3, 1, 1)``, reshape the per-channel stats in place (the ``_full`` copy from step (a) preserves the original).


Model and weight preparation
----------------------------

Download the LingBot-VA base checkpoint (Wan-Video 2.2 backbone, ~30 GB) into ``${LINGBOT_VA_MODEL_PATH}``. Expected layout::

   ${LINGBOT_VA_MODEL_PATH}/
   ├── transformer/                # 5B-param Wan transformer
   │   ├── config.json
   │   └── diffusion_pytorch_model-*.safetensors
   ├── vae/                        # Wan 2.2 VAE (not used during SFT)
   └── text_encoder/               # UMT5 (not used during SFT - we use a
                                     cached empty embedding)

Source: as provided in the original lingbot-va release notes.


Key LingBot-VA config fields
----------------------------

The single-GPU recipe in ``libero_sft_lingbotva.yaml`` is tuned tightly; the table below lists the load-bearing values and why they matter.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Setting
     - Value
     - Why
   * - ``actor.fsdp_config.strategy``
     - ``fsdp`` (FSDP1)
     - FSDP2 wraps parameters as DTensors, which crashes bitsandbytes ``AdamW8bit`` during warmup.
   * - ``actor.fsdp_config.sharding_strategy``
     - ``no_shard``
     - At world_size=1 sharding offers nothing. FSDP1 + NO_SHARD + ``use_orig_params=True`` keeps params as plain Tensors that bnb's CUDA kernels accept.
   * - ``actor.fsdp_config.use_orig_params``
     - ``True``
     - Same reason as above.
   * - ``actor.optim.optimizer_type``
     - ``adamw_8bit``
     - Required to fit optimizer state for the 5B-param transformer on a 45 GB card.
   * - ``actor.optim.lr`` / ``betas`` / ``wd``
     - ``1e-5`` / ``(0.9, 0.95)`` / ``0.1``
     - Match the reference recipe.
   * - ``actor.micro_batch_size``
     - ``1``
     - Memory limit on 45 GB.
   * - ``actor.global_batch_size``
     - ``10``
     - Implies ``gradient_accumulation = global / (micro * ws) = 10``.
   * - ``actor.model.lingbotva.attn_mode``
     - ``flex``
     - Reference recipe; requires torch ≥ 2.9. ``attn_mode: torch`` silently drops the BlockMask and collapses eval SR to 0 %.
   * - ``actor.model.precision``
     - ``bf16``
     - Transformer is loaded bf16 + force-cast (avoids fp32 leakage from ``scale_shift_table``, which would trip FSDP1's mixed-dtype check).
   * - ``runner.max_steps``
     - ``3000``
     - Reference run length.
   * - ``runner.save_interval``
     - ``1000``
     - First eval-able checkpoint at step 1000.


Launch training
---------------

Run from repository root:

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_lingbotva


Monitoring and sanity checks
----------------------------

1. Check ``run_embodiment.log``:

   - stable ``time/step`` (~17 s/step on L40)
   - reasonable ``train/latent_loss`` and ``train/action_loss``

2. Expected loss trajectory (means over the 10 grad-accum micro-batches of an optimizer step):

   .. list-table::
      :header-rows: 1
      :widths: 20 40 40

      * - Step
        - latent_loss
        - action_loss
      * - 1
        - ~0.20
        - ~0.55
      * - 50
        - ~0.18
        - ~0.16
      * - 100
        - ~0.15
        - ~0.14
      * - 500
        - ~0.13
        - ~0.12
      * - 1000
        - ~0.10
        - ~0.11

   Values within ±25 % are within typical RNG noise (CFG dropout + flow-matching timestep sampling diverge across torch versions even at the same seed). Values >2× off the reference at the same step count indicate something is wrong; the most common cause is ``attn_mode: torch`` slipping back in via a stale config or YAML merge — verify the active config logs show ``attn_mode: flex``.

3. TensorBoard:

   .. code:: bash

      tensorboard --logdir ./logs --port 6006


Evaluation
----------

Once ``checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt`` exists, evaluate with the standard RLinf entry point using the ``libero_object_eval_lingbotva`` config. The SFT checkpoint is selected via ``LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH``, and the 10 Libero-Object task ids are looped over via the Hydra CLI override ``env.eval.task_id_filter``.

.. code:: bash

   CKPT_DIR=${REPO_PATH}/runtime/lingbotva_sft/libero_sft_lingbotva/checkpoints
   export LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH=\
   ${CKPT_DIR}/global_step_1000/actor/model_state_dict/full_weights.pt
   OUT_ROOT=${REPO_PATH}/runtime/lingbotva_libero_eval/ckpt_1000

   for tid in 0 1 2 3 4 5 6 7 8 9; do
     bash examples/embodiment/eval_embodiment.sh libero_object_eval_lingbotva LIBERO \
       runner.logger.log_path=${OUT_ROOT}/task_${tid} \
       env.eval.total_num_envs=1 \
       env.eval.task_id_filter=[${tid}] \
       algorithm.eval_rollout_epoch=5
   done

Do **not** just set ``total_num_envs=50`` — the env's reset-state cursor cycles reset states within a task, not tasks. Expected mean SR at ckpt_1000 is ~26 % (range 18–28 % across seeds); below 10 % means ``flex_attention`` is not actually active and the model was silently trained under SDPA.


Common issues
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Symptom
     - Likely cause
     - Fix
   * - ``NameError: name 's10' is not defined`` during forward
     - torch < 2.9 ``flex_attention`` codegen bug
     - Upgrade to torch 2.9. Do **not** fall back to ``attn_mode: torch``.
   * - 0 % SR at ckpt_1000 despite normal-looking loss
     - SDPA mask leak — ``attn_mode`` is ``torch`` somewhere
     - Confirm the logged config shows ``attn_mode: flex``.
   * - ``ImportError: cannot import name 'flash_attn_func'``
     - torch 2.9 ABI breaks the flash_attn wheel
     - Apply the patch under "Environment" step 5.
   * - CUDA illegal memory access during warmup
     - FSDP2's DTensor + bnb AdamW8bit collision
     - Confirm ``fsdp_config.strategy: fsdp`` (FSDP1), not ``fsdp2``.
   * - FSDP1 "flatten tensors with uniform dtype" error
     - ``scale_shift_table`` initialised fp32
     - Confirm ``precision: bf16`` is set and the transformer is force-cast to bf16 (already wired in the action model).
   * - OOM during training
     - 45 GB card without ``expandable_segments``
     - Confirm ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is exported (``run_vla_sft.sh`` does this). Otherwise reduce ``num_action_chunks``.
   * - ``lerobot`` rejects dataset
     - ``episodes_stats.jsonl`` has wrong image-stats shape
     - Reshape per-channel stats from ``(3,)`` to ``(3, 1, 1)``.


Practical recommendations
-------------------------

- Run a short trial (e.g. 50–100 steps) after each config change to verify shapes, loss values, and throughput before committing to a multi-hour run.
- Multi-GPU scaling (world_size > 1) is **not yet verified** with this recipe. Keep FSDP1 (bnb AdamW8bit requires plain Tensors per rank), and verify the optimizer step before assuming the bnb + FULL_SHARD combination works. See ``reproduce.md`` §9 for additional pointers.
- The compute-bound forward/backward should give close-to-linear speedup once scaled, since data loading is only ~7 % of step time on L40.
