# Serving ten 20K-token task prompts with vLLM

Use ten application-managed task prompts and enable automatic prefix caching (APC) on the existing Gemma server. Each request includes the shared rules, the selected task's instructions, and the new user input. This gives each task the same token positions and attention visibility as a standalone request. The existing attention backend handles prefill and decode. The expected performance gain is lower time to first token on repeat requests. [Automatic prefix caching](../features/automatic_prefix_caching.md)

## 1. Build the request in the application

1. Store the shared rules and ten fixed instruction texts in a versioned prompt catalog. Use the application's existing task identifier when available. If the input is unlabeled, classify it before building the generation request. Handle unknown task IDs explicitly.
2. Use one request builder for warmup and live traffic. For a checkpoint whose template supports a system role, the message list can be built as follows:

   ```python
   def build_messages(task_id, user_text):
       instructions = TASK_PROMPTS[task_id]
       return [
           {"role": "system", "content": SHARED_RULES + "\n\n" + instructions},
           {"role": "user", "content": user_text},
       ]
   ```

3. Render representative requests with the deployed tokenizer and chat template. For each task, compare the token IDs from two requests with different inputs and record their longest common prefix. Use that measured length as the reuse target. Keep timestamps, request IDs, and per-request fields out of the fixed prefix. If the template requires a different role layout, preserve its supported ordering and repeat the token check.
4. Check all ten routes with distinct instruction fixtures. Assert that each rendered request contains the selected fixture and excludes the other nine. A direct standalone request for the same task and input should produce the same rendered token IDs.

## 2. Configure one serving replica

Record the actual model revision, vLLM version, GPU, tensor-parallel size, KV dtype, and effective attention backend. Enable APC on the existing serve command. A context limit of 32,768 is an example for a workload whose largest rendered input plus requested output fits within that limit.

```bash
vllm serve "$MODEL" --enable-prefix-caching --max-model-len 32768
```

Enforce the per-request token budget after templating. A fixed prefix of exactly 20,000 tokens leaves 12,768 tokens for all remaining input, template tokens, and output in this configuration. Send the complete selected prefix on every API call. vLLM matches resident blocks by token prefix. An early edit to an instruction invalidates reuse downstream of that edit. [Cache matching and eviction](prefix_caching.md)

## 3. Estimate the KV working set

For ten distinct prefixes of exactly 20,000 tokens, the following estimate assumes the public Gemma 4 **12B IT** configuration, one GPU, BF16 K and V stored separately, and one 1,024-token local window retained per branch. Check the deployed checkpoint behind the server's model name before applying these numbers. [Gemma configuration](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json) · [Hybrid KV allocation](hybrid_kv_cache_manager.md)

| Layer group | Bytes for one branch | One branch |
| --- | --- | ---: |
| Global | 8 layers × 20,000 tokens × 1 KV head × 512 elements × 2 tensors × 2 bytes | 312.5 MiB |
| Sliding window | 40 layers × 1,024 tokens × 8 KV heads × 256 elements × 2 tensors × 2 bytes | 320 MiB |

The raw estimate is **632.5 MiB per branch, or 6.18 GiB for ten** before common-prefix sharing. Actual allocation depends on cache grouping, page rounding, retained blocks, and active suffixes. Budget separately for weights and runtime allocations. Record the KV capacity reported at startup and verify residency under realistic concurrency.

## 4. Measure reuse before rollout

| Run | Procedure | Evidence to collect |
| --- | --- | --- |
| Cold baseline and warmup | On a fresh, isolated replica, send one request per task with a short user input and a one-token output limit. | Per-task time to first token (TTFT) and rendered prompt length. Shared rules may already hit after the first task. |
| Warm check | Repeat the same ten tasks sequentially against that replica. Change the first user-content tokens for each request. Snapshot the replica's prefix-hit counter before and after each request. | Compare each hit-token delta with the measured common prefix and the cache block granularity. Ten 20K reusable prefixes should yield around 200K hit tokens in total. Inspect each task as well as the sum. |
| Load check | Repeat with production-like concurrency, task frequencies, user input lengths, and output limits. Interleave all ten tasks and revisit the least frequent ones. | Compare warm and cold TTFT, report p50/p95 warm TTFT, and track cache hits, preemptions, and throughput. New user tokens still incur prefill work. |

Use the token-valued `vllm:prefix_cache_hits` counter exposed by the installed version at `/metrics` (Prometheus may append `_total`). Read the same engine's labeled series for each delta. Run the sequential test without other traffic so the deltas belong to the requests being measured. [Production metrics](../usage/metrics.md)

## 5. Roll out and handle cache misses

Canary the application router on one replica once every route passes the isolation check, every prefix shows the expected warm reuse, and measured latency meets the service's target at expected load. If predictable first-request latency matters, repeat warmup after restarts and prompt revisions. With several independent replicas, warm the branches each replica serves or route a task consistently to the replica holding its prefix.

If warm hits are short even on the isolated replica, compare actual token IDs, confirm both requests reached the same engine, and check APC settings. If hits fall only under load, vary concurrency and KV capacity on the canary and repeat the interleaved test. Cache eviction leads to ordinary prefill on a miss. Roll back the application routing or server configuration independently if the canary misses the service target.
