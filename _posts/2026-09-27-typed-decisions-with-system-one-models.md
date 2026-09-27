---
layout: post
title: "Typed Decisions Instead of Text: Jev and Two Open-Weight System One Models"
date: 2026-09-27
description: "A System One model answers typed questions in one forward pass and returns calibrated probabilities instead of prose. How Jev, Laya, and Decider work, traced through the weights."
tags: [system-one-models, jev, machine-learning, agents, calibration]
---

If you have dealt with agents where the model was asked to classify something, you would have seen at times it answers in prose. You need a wrapper code with a regex or a JSON parser downstream has to turn that prose back into a parsed structure. In addition, the model was never asked for a probability, so the code has no way to differentiate a confident answer from a guess. It just takes the string.

To address this, a different class of model has appeared. You send three inputs to the model. State, a list of typed questions, the expected format/ options for the response. The model conforms to the structure you specified along with with probabilities, most importantly in a single forward pass, without generating any text. TypeSafe AI calls the category a System One model and ships the first public one, Jev [1]. If you are familiar with System one, its name borrows Kahneman's split between fast automatic judgment (System 1) and slow deliberate reasoning (System 2)[2]. LLM does the reasoning, and the new class of models do the automatic judgments.

This post covers what these models return, why the architecture matters for how you use them, and how one of them works well enough that you could write the inference loop yourself. Jev's architecture is not published, so the architecture section uses two open-weight models instead: Laya, an encoder, and Decider, a decoder.

Interesting use cases are emerging how to apply this. In the last section, I demonstrate how this can be wired with a Strands agent as a steering handler.

Code: [github.com/GopiKWork/jev-experiments](https://github.com/GopiKWork/jev-experiments)

## About the Setup

If you want to follow the code, everything measured here ran on one machine: an 8-core aarch64 CPU with 30 GB of RAM, no GPU, Amazon Linux 2023, Python 3.12. Package versions are `typesafe-sdk` 0.7.1, `laya` 0.3.20, `decider-ai` 1.5.0, and `strands-agents` 1.57.0, all installed with `uv` into one virtual environment.

As of this writing, Jev is maxed out on the API Keys. However you can access it via OpenRouter [4], so no TypeSafe key is needed. The Strands agent and the LLM steering baseline both use OpenAI's GPT-5.6 Terra on Amazon Bedrock. I tested Laya and Decider run locally on the CPU, which is slow for Decider and matters for the latency numbers later.

The running example is a car dealership assistant that books service appointments. One customer message is used throughout:

> When I stop at lights the car pulls hard to the left and the pedal sinks almost to the floor.

That sentence describes failing brakes, and it contains no word that a keyword rule would catch. No "brake", no "grinding", no "smoke". I picked it because it separates the methods cleanly.

## What Jev Returns

TypeSafe's docs define three question primitives, and every request is some combination of them [3]:

| Type | The question it asks | What comes back |
|---|---|---|
| Choice | Pick one option from a list | the winning label, a probability per label, a confidence |
| Score | Place the state on an ordered rubric | a float, a confidence, a legend, a probability per level |
| Noul | Is this statement true? | one probability between 0 and 1 |

The docs note that every question in a call is evaluated in parallel and in isolation against the same state, and that adding questions barely affects latency [3]. This is due to the architecture and we will get to that below.

It should be noted that Confidence is not a single quantity across these three implementations. Jev and Decider return a `confidence` derived from the spread of the probabilities. Laya returns two fields, a `confidence` that is a normalised value (entropy) and an `answer_confidence` that is simply `max(p)` over the valid options. If you compare confidences across models you will be comparing different statistics.

Here is the whole call for JEV with OpenRouter as the hosting platform:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api")

response = client.system_one(
    model="~typesafe/jev-latest",
    state=MESSAGE,
    questions={
        "department": Choice(
            instructions="Which dealership department should handle this?",
            criteria={"Service": None, "Parts": None, "Sales": None, "Warranty Claims": None},
        ),
        "urgency": Score(
            instructions="How soon does this need attention?",
            criteria=["Routine", "Within a week", "Immediately"],
        ),
        "safety_risk": Noul(instructions="Is it unsafe to keep driving this vehicle?"),
    },
)
```

As you can see from above all three questions are covered in a single HTTP request. There is no output schema, no instruction to reply in JSON, no temperature, and no retry for when the model answers in a sentence instead of a label. `criteria` is a dict with labels as keys and optional descriptions as values. The option set lives in your code as data instead of a substring in a prompt template.

Response can be parsed from the response directly.

```python
answers = response.answers
safety_risk = answers["safety_risk"].noul
urgency = answers["urgency"].score

priority = safety_risk >= 0.6 or urgency >= 1.5
```

Output:

```
department:  Service (1.00)
urgency:     Immediately (2.00)
safety_risk: 0.94
```

In the above example, the policy for priority is a combination of two thresholds. Key is you can read them, unit test them, and change a threshold (0.6 to 0.5, say) without touching a prompt.

![Where a System One model sits next to an LLM in an agent loop](/assets/images/jev-experiments/figure-1-agent-loop.png)

*Figure 1. Integration between LLM (System Two) and Jev (System One)*

## Two Open-Weight Options

As of this writing (Sep 2026), TypeSafe has not published Jev's architecture[1]. So if you want to know how a model can answer typed questions in one pass, you have to look at something you can download. I have explored two Apache 2.0 checkpoints. Interestingly, they target the same interface from opposite directions.

**Laya** [5] describes itself as a "Multilingual, non-autoregressive System 1 decision model" and its checkpoint is a fully fine-tuned ModernBERT-large backbone with a decision head trained from scratch: 421M parameters, 512-token context. ModernBERT is an encoder and doesn't produce a next token.

**Decider** [6] Per their docs, this is a language model that does not generate text. Its `decider-4b` checkpoint is fine-tuned from Qwen3.5-4B-Base [7]: 4.2B parameters, 32 layers, 8 of them full attention (hidden size 2,560, 8.4 GB of bf16 weights). You can check the notebook in the Github repo to explore the weights further.

However the two solve the same problem in different ways and employ different tricks.

| | Laya (encoder) | Decider (decoder) |
|---|---|---|
| Backbone | ModernBERT-large, bidirectional | Qwen3.5-4B-Base, causal |
| Parameters | 421M | 4.2B |
| Where options live | inline in the sequence, as their own text | rewritten as letters `(A) (B) (C)` in a prompt |
| Where the answer is read | one position per option | one position per question |
| Decision parameters added | 26.5M, or 6.3% of the model | zero |
| Weights on disk | 0.8 GB | 8.4 GB |

Laya puts a `[MASK]` token in front of every option and reads one scalar out of each mask position. I read its head definition out of the installed package:

```python
self.scorer   = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
```

The output width of `scorer` is 1. That single number is "how well does this one option fit?", and because the same scorer is applied once per mask, a question can have two options or forty without changing a weight. This is the part that differs from the BERT classifier you may have fine-tuned before, where the head is a `Linear(d, num_classes)` and the class list is frozen into the shape. Adding options is free in parameters but not in resolution: the options share a fixed prompt budget, so past roughly twenty labels each one gets truncated to a few tokens and they stop being distinguishable. The model card measures the cost on a 77-label task, where accuracy falls to 0.425 against Jev's 0.870 [5].

Decider does something stranger, and that is the rest of this section.

![Encoder and decoder readouts side by side](/assets/images/jev-experiments/figure-2-encoder-vs-decoder.png)

*Figure 2. Where each model reads its answer from: one position per option in Laya, one position per question in Decider*

The Laya row in that figure needs a line of explanation, because its numbers are not self-evident. The four scalars, `0.262`, `-0.369`, `-1.917` and `-1.196`, are the raw `scorer` outputs, one per `[MASK]`, in option order. They are not probabilities and they are not bounded. They are just fit values, and on their own they are only comparable to each other.

An ordinary softmax turns them into probabilities, but Laya divides them by a constant first, and that constant is where `1.76` comes from. It is not in the prompt and it is not computed per request. It is a calibration temperature that ships inside the checkpoint, in `rl_agent_config.json`, under a map keyed by question type and option count:

```json
"temperature_by_options": {
  "choice:2":    1.9063563346862793,
  "choice:3-5":  1.7601518630981445,
  "choice:6-10": 1.0000158548355103,
  "choice:11+":  0.10058280825614929,
  "score:3-5":   1.2514300346374512,
  "noul:2":      1.983399510383606
}
```

The department question has four options, and `laya/common.py` buckets four as `3-5`:

```python
def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)
```

So the divisor is the `choice:3-5` entry, 1.7601518630981445, which is the 1.76 in the figure rounded for the diagram. Dividing by it flattens the four scalars into `Service 0.41`, `Parts 0.29`, `Sales 0.12`, `Warranty Claims 0.18`, which is what `laya.predict` returns for this sentence. Two details are worth knowing if you rely on these numbers. A missing key falls back to a three-element per-type list, so only six of the possible buckets are actually fitted here. And any value outside [0.5, 5.0] is clamped at load with a warning, which the shipped `choice:11+` entry of 0.1006 trips: a temperature that low sharpens instead of softening, and would publish a near coin flip as a near certainty. The next section explains what a calibration temperature is and how a number like this gets chosen, since Decider uses one too.

Worth noticing that the winner here takes only 0.41. Laya routes this message correctly, but it is not confident about it, and a policy with a 0.5 floor would reject its answer.

## Reading a Decision Out of a Decoder

A decoder has exactly one head, the language-model head, and it produces a distribution over the whole vocabulary. Decider does not add a second head. Start with the prompt it actually builds. This is the exact string sent to the model for the Choice question, decoded back from the token ids:

```
Context:
When I stop at lights the car pulls hard to the left and the pedal sinks almost to the floor.

Question: Which dealership department should handle this?
Options:
(A) Service
(B) Parts
(C) Sales
(D) Warranty Claims
Answer: (
```

In this case it is 58 tokens, ending on an open parenthesis. However the input is kept as multiples of 64 tokens and padded up. The last four token ids are `[198, 15666, 25, 318]`, which decode to `'\n'`, `'Answer'`, `':'`, and `' ('`. That final `' ('` is token index 57, and the code records it as the answer slot when it builds the prompt:

```python
slots.append(len(ids) - 1)     # position of " (" token
```

Think about what an ordinary decoder would do here. Given a prompt that ends in `Answer: (`, the very next token is likely to be `A`, `B`, `C`, or `D`, because nothing else fits the pattern the prompt set up. The model has already done the classification by the time it reaches that parenthesis. A generating model would sample one token and hand you a string. However the Decider skips the sampling and reads the distribution.

Here is the whole read, seven lines from `decider/model.py`:

```python
def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts):
    h = self.lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    hs = h[slot_batch, slot_idx]                                   # [N,H]
    W = self.lm.lm_head.weight[self.letters]                       # [K,H]
    logits = F.linear(hs, W).float()                               # [N,K]
    ar = torch.arange(MAX_OPTIONS, device=logits.device)[None, :]
    logits = logits.masked_fill(ar >= nopts[:, None], float("-inf"))
    return logits
```

Line by line, with the shapes from the traced run:

`self.lm.model(...)` calls the backbone and not the model. `self.lm` is the full `Qwen3_5ForCausalLM`; `self.lm.model` is everything except the output head. It returns `last_hidden_state` of shape `(1, 64, 2560)` in bf16: one 2,560-dimensional vector per token position. 64 rather than 58 because the batch collate pads up to a multiple of 64.

`h[slot_batch, slot_idx]` is fancy indexing, the same thing as `h[0, 57]` for a single row (single batch item), and it throws away 63 of the 64 positions. What is left is `(1, 2560)`, the model's internal state at the open parenthesis.

`self.lm.lm_head.weight[self.letters]` is the trick. The full head is a `(248320, 2560)` matrix, one row per vocabulary token. `self.letters` is a tensor of 255 token ids, and its first six are `[32, 33, 34, 35, 36, 37]`, which decode to `A`, `B`, `C`, `D`, `E`, `F`. Indexing with it slices out a `(255, 2560)`. Everything else in the vocabulary is simply ignored.

`F.linear(hs, W)` is a matrix multiply producing `(1, 255)`. Each entry is the dot product of the slot's hidden state with one letter's embedding, which is exactly the logit the full head would have assigned that letter. Nothing new is computed; 99.897% of the head's work is skipped.

`masked_fill(ar >= nopts[:, None], -inf)` sets every row past the option count to negative infinity. With four departments, the raw logits `[14.25, 5.19, 4.88, 11.50, 5.84, 5.19, ...]` become `[14.25, 5.19, 4.88, 11.50, -inf, -inf, ...]`, leaving 4 finite entries out of 255. After a softmax, `-inf` contributes exactly zero, so the probabilities sum to 1 over the real options and the model cannot answer `E`.

One softmax later, with the temperature from the checkpoint's `decider_config.json`:

```
                 raw softmax    at T=1.935
Service              0.9397        0.7946
Parts                0.0001        0.0073
Sales                0.0001        0.0063
Warranty Claims      0.0601        0.1918
```

Those right-hand numbers match `decider.system_one()` exactly, key for key, which is the check that the hand trace is the real path and not a plausible story about it. The trace is in `trace_decider.py` in the companion repo; it rebuilds the rows through Decider's own `_system_one_items`, runs the backbone itself, and prints every figure quoted in this section next to the value the API returns.

![The Choice read path, from prompt to probabilities](/assets/images/jev-experiments/figure-3-choice-read-path.png)

*Figure 3. The Choice read path, from prompt to probabilities*

The masking is doing real work. Let the same hidden state address the full 248,320-token vocabulary instead, at T=1 so the numbers line up with the left column: `'A'` gets 0.7975 and `'D'` gets 0.0510, and the four valid letters together get 0.8486. The other 15.14% goes to tokens that are not options. Some of it is the same answer in a different costume, and those are the ones worth looking at, because a string parser would have to handle every one: `' A'` with a leading space at 0.0045, lowercase `'a'` at 0.0025, `'.A'` at 0.0004, the Cyrillic `'А'` at 0.0005, `'(A'` at 0.0003. But they add up to well under a point. The bulk of the 15% is not near-misses at all, it is a long diffuse tail across thousands of unrelated tokens, each with a tiny share. That is the more useful observation: the distribution a decoder produces is only approximately about your question, and restricting it to 255 known rows before the softmax is what turns it into numbers you can threshold.

## What the Calibration Temperature Is

I have quoted T = 1.935 in the table above and in Figure 2 without saying what it is, and it deserves a section, because it is the only number in the whole read that was not learned during training.

Look at the raw softmax column again. Service gets 0.9397. The question a calibration temperature asks is whether that 0.94 is honest: across many decisions where this model reports 0.94, is it actually right about 94% of the time? For most trained networks the answer is no. They are systematically overconfident, and their reported probability sits above their observed accuracy. If you plan to compare a probability against a threshold in code, that gap is your problem, not the model's.

The fix is temperature scaling, and it is deliberately the smallest fix available. Freeze every weight in the model. Add exactly one scalar, T. Divide the logits by it before the softmax:

```python
p = softmax(logits / T)
```

Then choose T by fitting it on a held-out labelled set: search for the single value that best lines the reported probabilities up with the observed accuracy, optimising over T alone while the network stays frozen. That is the whole procedure. One parameter, fitted after training, on data the model did not train on. The loss you fit against is a choice, and the two checkpoints here make different ones. Negative log-likelihood of the true answers is the textbook default. Laya's card instead reports expected calibration error, measured on `max(p)`, and quotes the improvement directly: refitting one temperature per question type and option count moves mean ECE from 0.466 to 0.081 [5].

Two properties make it safe. T > 1 flattens the distribution and T < 1 sharpens it, so an overconfident model gets a T above 1. And dividing every logit by the same positive number is monotone, which means the ranking cannot change. Calibration never changes which option wins. It only changes how strongly the model claims it.

For Decider, T came out at 1.935, fitted on 61 of the 67 in-task regression tasks in its training mixture [6]. That is what drops Service from 0.9397 to 0.7946. Service still wins, and 0.79 is the number your threshold should be compared against. Laya's 1.76 is the same mechanism, one scalar dividing the logits, differing only in bookkeeping: Laya stores a map and picks an entry per question type and option-count bucket, while the `v2` Decider tag stores one number and uses it everywhere.

Which points at the honest limitation. One temperature does not fit every task type: the `v2` tag I pinned uses a single T across Choice, Score and Noul, while the newer `v2.1` default ships a per-type map [6]. The card's own advice is the right advice. If you route on confidence, refit the temperature on your own labels.

## The Same Slot, Three Answer Types

The interesting part is that only the last few lines differ between the three types. Everything above is shared.

We covered Choice already. The prompt lists the options as letters, softmaxes over `nopts` entries, and the answer is `names[argmax(p)]` with the full probability dict alongside.

Noul is a Choice with two options. It gets its own branch in the readout. The question renderer builds this:

```
Question: Is it unsafe to keep driving this vehicle?
Options:
(A) no
(B) yes
Answer: (
```

Score returns a number. By default the shipped config sets `isolated_levels: true`, which means a three-level rubric becomes three separate yes/no prompts, one per level:

```
Question: How soon does this need attention?
Proposed answer: Within a week
Does the proposed answer fit?
Options:
(A) no
(B) yes
Answer: (
```

Three prompts, batched into one forward pass. The per-level probability of "yes" came out as 0.0311 for Routine, 0.2501 for Within a week, and 0.6035 for Immediately. Two lines of arithmetic finish it. First normalise:

```python
def combine_isolated(p_yes):
    tot = sum(p_yes) or 1e-9
    return [x / tot for x in p_yes], tot
```

`tot` is 0.8846, and the normalised weights are `[0.0351, 0.2827, 0.6822]`. Then take the expectation over level indices:

```python
"score": round(sum(i * x for i, x in enumerate(p)), 2)
```

which is `0 x 0.0351 + 1 x 0.2827 + 2 x 0.6822 = 1.6471`, reported as 1.65. That is why a Score answer is a float between the level indices rather than one of them: 1.65 means "between Within a week and Immediately, leaning hard toward Immediately". To get a label back, round it and look it up in the returned legend, remembering that Decider's legend keys are strings, so it is `legend[str(round(score))]` and not `legend[round(score)]`.

You can turn isolation off and get one prompt with the levels as options `(A) 0: Routine`, `(B) 1: Within a week`, `(C) 2: Immediately`. On this same sentence that path gave probabilities `[0.0493, 0.1577, 0.7930]` and a score of 1.74 rather than 1.65. The isolated form is the default because each level is judged without its number and without its neighbours, so adding or removing a level cannot move the others. I have not measured which is more accurate on a labelled set, but the model card has: across five rating datasets it changes accuracy by at most 2.0 points [6].

![One backbone, three readouts](/assets/images/jev-experiments/figure-4-three-readouts.png)

*Figure 4. One backbone, three answer types: only the final arithmetic differs*

## Applied: A Third Option for Strands Steering

One place where System one models can be applied in Harness is steering. Take Strands for example. It has a steering mechanism for exactly the class of problem where this fits. A steering handler watches the agent and can intervene before a tool call, returning `Proceed` to let it run, `Guide` to cancel it and send feedback, or `Interrupt` to pause for a human [8]. The framing in the docs is modular prompting: instead of one large prompt carrying every rule, guidance appears when it is relevant. In the Python SDK the registered hook callback is `provide_tool_steering_guidance`, which fires on `BeforeToolCallEvent`; the method you override in your own handler is `steer_before_tool`, which receives the agent and the `tool_use` and returns one of those actions [9]. There is a second tap after the model responds, `steer_after_model`, where only `Proceed` and `Guide` are available. `Interrupt` is a tool-time action only, a point the handler's own module docstring gets wrong.

The scenario is a dealership agent with a deliberately careless system prompt. It books everything at standard priority and never asks a follow-up question. Policy says a Service booking for an unsafe vehicle must be same-day. Steering has to catch that before the booking lands.

Until now there were two ways to write the handler. Deterministic Python is fast and testable and only as good as its keyword lists. A second LLM call reads the policy in natural language, which handles the language but costs a round trip. A System One model gives a third shape: the model answers the questions, and Python owns the policy.

```python
class DeciderSteering(SteeringHandler):
    THRESHOLD = 0.6

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        answers = decider_agent().system_one(
            customer_text(agent),
            {
                "safety_risk": {"type": "noul",
                                "instructions": "Is it unsafe to keep driving this vehicle?"},
                "department": {"type": "choice",
                               "instructions": "Which dealership department should handle this?",
                               "criteria": DEPARTMENTS},
            },
        )["answers"]

        same_day = (answers["department"]["choice"] == "Service"
                    and answers["safety_risk"]["noul"] >= self.THRESHOLD)
        return check(tool_use["input"], answers["department"]["choice"], same_day)
```

`check` compares the proposed booking against the expected one and returns `Guide` with a corrective sentence or `Proceed`. The policy, the threshold, and the comparison stay in one place you can test without a model.

![Jev steering the dealership agent across two passes](/assets/images/jev-experiments/strands-steering-jev-walkthrough.png)

*Figure 5. Jev steering the dealership agent across two passes, from proposed booking to corrected priority*

## What the Five Methods Measure

`scenarios.jsonl` holds 50 dealership messages, each labelled with the booking correct steering should produce. A scenario counts as correct only when both the department and the priority match. The set is adversarial on purpose: 19 messages describe an unsafe vehicle, many without a safety keyword, and several safe messages do contain one, such as a peeling steering wheel or brake pads a customer wants to fit themselves.

One full run over all 50 scenarios, 8 at a time, on the CPU described above, on 25 September 2026. `Avg` is wall clock per scenario end to end: it includes the agent's own Bedrock calls and any steering retry, not just the steering decision. Model load happens once up front and is excluded.

| Method | Accuracy | Total (s) | Avg (s) | Network call | Weights |
|---|---|---|---|---|---|
| `deterministic` | 56% | 10.9 | 1.6 | No | None |
| `laya` | 72% | 35.0 | 5.2 | No | 0.8 GB |
| `llm` | 80% | 19.9 | 3.0 | Yes | None |
| `decider` | 94% | 174.5 | 26.5 | No | 8.4 GB |
| `jev` | 98% | 14.5 | 2.1 | Yes | None |

The ranking is the useful result, not the exact percentages, since this is one run on 50 hand-written scenarios of my own. Keyword rules miss the hazards that use no keyword and fire on the safe messages that do. The LLM judge fixes most of the routing and still waves unsafe vehicles through at standard priority, and its set of misses changes between runs, which is its own argument. Jev missed one scenario out of 50: a burnt-out brake light bulb scored 0.67 safety risk against my 0.6 threshold, which is a threshold problem rather than a classification failure. Decider came within 4 points of it with no network call and no API key.

The two local methods have inflated averages. Timed alone, a Laya prediction takes about 0.5 seconds and a Decider prediction about 3 seconds on 8 CPU cores. With `--workers 1`, Laya averages 2.2 seconds per scenario against Jev's 2.0. Decider would be much faster on a GPU, where it can use the Triton linear-attention kernel instead of the reference PyTorch path I had to fall back to.

![Accuracy against cost for the five steering methods](/assets/images/jev-experiments/figure-6-accuracy-vs-latency.png)

*Figure 6. Accuracy against measured average latency for the five steering methods*

## Conclusion

A System One model replaces a parse with a probability. The interface is three primitives, Choice, Score and Noul, and the value is that the option set and the thresholds live in your code rather than inside a prompt. Jev's architecture is unpublished, so the two open-weight checkpoints are where the mechanism is visible: Laya scores each option at its own `[MASK]`, and Decider reads one hidden state at an open parenthesis and scores option letters against it. All three question types share that single read. Only the final arithmetic differs, and a Score is an expectation your own code computes.

The rule for when to reach for one: the judgment is linguistic, but the policy is not. "Is this vehicle unsafe to drive" needs language. "Unsafe Service bookings go same-day" does not. That second half is an `if` statement, and it belongs somewhere you can unit test it.

**Where this fits in coding agents**

That rule describes a surprising amount of the plumbing inside a coding agent, where most decisions are small, repeated, and currently made either by a brittle regex or by a full LLM call that costs a round trip and returns prose.

| Decision | Question type | What it replaces |
|---|---|---|
| Issue triage and routing | Choice over labels, owners or repository paths | a keyword rule, or an LLM call per issue |
| Tool-call gating | Noul: does this command need human approval? | a hand-maintained deny list |
| Retrieval and reranking | Score per retrieved chunk against a relevance rubric | a bare similarity cutoff |
| Diff and output verification | Noul or Score against rubric constraints, before the harness applies a patch | an LLM judge, or nothing |
| Loop control: retry, stop, escalate | Choice, with abstention as a real answer | a fixed iteration cap |

The last row is the one I find most interesting, because a calibrated probability is exactly what a retry loop has always been missing. "Stop when the model says it is done" is unreliable, and "stop after five attempts" is arbitrary. "Stop when p(resolved) is above 0.9, escalate below 0.4, retry in between" is a policy with numbers in it, which means you can tune it against a labelled set and watch it change.

**Automation: Jev in a request path**

KiroCrew already uses Jev this way. Skill routing is a Choice question: given the incoming message and the names and descriptions of the installed skills, which skill fits, or none of them. Abstention is a real answer rather than a low score to be thresholded later. If Jev abstains, is slow, or is unreachable, a word-matching rule takes the message instead, so a decision never holds up a reply. It applies to a configurable share of sessions, every call is logged locally, and each category of state that leaves the machine has its own switch that defaults to off.

**What I expect next**

Two ways this could go. The capability folds into general language models as another decoding mode, and `system_one` becomes a parameter on a chat completion. Or System One stays a separate, small, cheap class of model that sits beside the LLM and answers the questions the LLM should not be spending tokens on. On cost alone the second looks more likely: the entire point is that a typed decision is one forward pass of a 4B model with no generation, and folding it into a frontier model gives that up. Either way the interface is the durable part. Typed questions in, calibrated probabilities out, policy in code.

## References

[1] TypeSafe AI. "TypeSafe AI." typesafe.ai, 2026. [https://typesafe.ai/](https://typesafe.ai/)

[2] D. Kahneman. "Thinking, Fast and Slow." Farrar, Straus and Giroux, 2011.

[3] TypeSafe AI. "Introduction." TypeSafe AI documentation, 2026. [https://docs.typesafe.ai/](https://docs.typesafe.ai/)

[4] OpenRouter. "Model record for ~typesafe/jev-latest." OpenRouter API, 2026. [https://openrouter.ai/api/v1/models/~typesafe/jev-latest/endpoints](https://openrouter.ai/api/v1/models/~typesafe/jev-latest/endpoints)

[5] Convai Innovations. "convaiinnovations/laya." Hugging Face, 2026. [https://huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)

[6] Mapika. "Mapika/decider-4b." Hugging Face, 2026. [https://huggingface.co/Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b)

[7] Qwen. "Qwen/Qwen3.5-4B-Base." Hugging Face, 2026. [https://huggingface.co/Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)

[8] Strands Agents. "Steering." Strands Agents documentation, 2026. [https://strandsagents.com/docs/user-guide/sdk/agents/interventions/steering/](https://strandsagents.com/docs/user-guide/sdk/agents/interventions/steering/)

[9] Strands Agents. "strands.vended_plugins.steering.core.handler." Strands Agents API reference, 2026. [https://strandsagents.com/docs/api/python/strands.vended_plugins.steering.core.handler/](https://strandsagents.com/docs/api/python/strands.vended_plugins.steering.core.handler/)

---

*Thank you for taking the time to read and engage with this article. Your support in the form of following me and sharing the article is highly valued and appreciated. The views expressed in this article are my own and do not necessarily represent the views of my employer. If you have any feedback and topics you want to cover, please reach me at [LinkedIn](https://www.linkedin.com/in/gopinathk/)*
