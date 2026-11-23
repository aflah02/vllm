# Labeling every input against ten 20K-token moderation policies

For each incoming input, evaluate all ten policies and return one label per policy. Submit ten independent generation requests to the existing vLLM service. Each request contains the common evaluation instructions, one fixed policy, and the same input text. This gives each evaluation the context of its standalone policy run and allows vLLM to reuse that policy's KV prefix across different inputs. The existing FA3/FA4 serving path can perform these evaluations. [Automatic prefix caching](../features/automatic_prefix_caching.md)

## 1. Expand an input into ten evaluations

- Store the ten policies in a versioned catalog with stable IDs and the allowed labels for each policy. Snapshot the policy revisions and evaluator configuration when admitting an input so an update midway through processing cannot mix versions.
- Create one pending evaluation per policy. Key each by immutable input revision, policy ID, policy revision, and evaluator version. Use the same key for retries. A uniqueness constraint or an equivalent idempotent write prevents duplicate deliveries from adding extra labels.
- For each evaluation, render the same shared instructions and policy text followed by the input. Keep correlation IDs and timestamps in application metadata. Use one request builder for both warmup and live traffic. For a chat template that supports the system role, the messages can be built as follows:

  ```python
  def build_messages(policy_text, input_text):
      return [
          {"role": "system", "content": SHARED_RULES + "\n\n" + policy_text},
          {"role": "user", "content": input_text},
      ]
  ```

- Generate a short label from that policy's allowed set. Set the output limit high enough for its longest valid label in the deployed tokenizer. Validate the result and persist it under the evaluation key. Return a per-policy label map when every expected policy has a valid result. On a transient failure, retry only the outstanding evaluation. Surface a partial or failed status if any evaluation exhausts its retry budget.

## 2. Bound concurrency and reuse the prefixes

Use a queue and a shared limit on in-flight evaluations per serving replica. For example, 100 inputs produce 1,000 evaluations that can be drained over time. Submit independent requests concurrently so vLLM can batch work across inputs. Tune the limit against completed inputs per second and the time until the tenth label arrives. Bound the queue or apply backpressure when demand exceeds that measured rate.

Enable APC on the existing serve command. Set the context budget for one rendered policy, one input, and its output. For example, 32,768 tokens allows 12,768 additional tokens after an exactly 20,000-token prefix.

```bash
vllm serve "$MODEL" --enable-prefix-caching --max-model-len 32768
```

Warm each policy once on the replica that will serve it. Send the full policy and current input on every subsequent request. Resident KV blocks can skip most policy prefill. The input text follows a different prefix in each evaluation, so it incurs prefill ten times. Each generation also attends to its policy context. For a 500-token input, budget roughly 5,000 newly prefilled input tokens per completed input on a warm server, plus template overhead, unmatched block tails, and the ten outputs. [Cache matching and eviction](prefix_caching.md)

Start with all ten policies on one replica if they stay warm at target concurrency. With several replicas, assign policy sets to replicas and dispatch each evaluation to one that holds its policy. Replicate a set if its queue becomes the bottleneck. If ten prefixes cycle out on a single GPU and requests can tolerate queueing, benchmark short same-policy batches across inputs. Bound the wait for each policy so one missing label does not hold completed inputs indefinitely.

## 3. Estimate the KV working set

For ten distinct prefixes of exactly 20,000 tokens, the following estimate assumes the public Gemma 4 **12B IT** configuration, one GPU, BF16 K and V stored separately, and one 1,024-token local window retained per policy. Check the deployed checkpoint behind the server's model name before applying these numbers. [Gemma configuration](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json) · [Hybrid KV allocation](hybrid_kv_cache_manager.md)

| Layer group | Bytes for one policy | One policy |
| --- | --- | ---: |
| Global | 8 layers × 20,000 tokens × 1 KV head × 512 elements × 2 tensors × 2 bytes | 312.5 MiB |
| Sliding window | 40 layers × 1,024 tokens × 8 KV heads × 256 elements × 2 tensors × 2 bytes | 320 MiB |

The raw estimate is **632.5 MiB per policy, or 6.18 GiB for ten** before common-prefix sharing. Actual allocation depends on cache grouping, page rounding, retained blocks, and active suffixes. Budget separately for weights and runtime allocations. Record the KV capacity reported at startup and verify residency under realistic concurrency.

## 4. Validate correctness, reuse, and throughput

| Check | Procedure | Acceptance evidence |
| --- | --- | --- |
| Policy coverage and correctness | Run labeled fixtures including an input with different expected outcomes under two policies. Compare each rendered request and prediction with a standalone run of the same policy. Test an unknown label, a duplicate delivery, and a single failed evaluation followed by a retry. | Exactly ten distinct valid policy results per completed input. A missing label keeps the input incomplete. Assess agreement and moderation quality separately for each policy. |
| Warm prefix reuse | On an idle replica, send one input through all ten policies to warm them. Send a second, different input through all ten. Run the second pass serially to attribute each prefix-hit counter delta to one policy. | Compare hits per policy to the tokenized common prefix of the two rendered requests. Expect about 200K cached tokens in total for ten 20K resident prefixes, subject to block boundaries. |
| Production load | Use representative arrival rates, input lengths, and label lengths. Increase the in-flight limit in steps. Measure from input admission to persistence of the tenth valid label, including time in the queue and retries. | Sustainable **fully labeled inputs/second**, p50/p95 completion latency, queue depth, retry rate, preemptions, and cache hits. At 10 inputs/second, the service must sustain at least 100 successful policy evaluations/second. |

Read the token-valued `vllm:prefix_cache_hits` counter for the target engine from `/metrics` (the exposed series may have a `_total` suffix). Compare warm and cold time to first token to diagnose prefill savings. Use end-to-end completion latency to decide whether the moderation service meets its target. [Production metrics](../usage/metrics.md)

## 5. Roll out

Run the ten-policy evaluator on a representative labeled sample, then canary it at realistic traffic. Record checkpoint revision, vLLM version, GPU, tensor parallelism, KV dtype, and per-replica cache capacity alongside the results. Increase traffic once policy coverage, label quality, throughput, and completion latency meet the service targets. On a policy revision, warm its new prefix and use the snapshotted revision for in-flight evaluations.
