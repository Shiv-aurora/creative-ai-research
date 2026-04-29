# Creativity as a measurable capability in LLMs

## Why “creativity” is hard to measure but still researchable

Most serious work converges on a simple idea: **creativity is not just novelty**. In creativity research, the “standard definition” is typically **bipartite**: creative outputs must be **original** *and* **effective/appropriate** (i.e., they have to work). citeturn10search1 This matters because many LLM “creativity wins” come from metrics that mostly reward novelty—sometimes rewarding nonsense.

A newer thread in creativity science argues that **product-only definitions** (novel + useful) don’t fully capture the phenomenon, because creativity is also a **process**. One peer‑reviewed process definition proposes: creativity is “**internal attention constrained by a generative goal**,” and explicitly distinguishes *creativity-as-process* from *creative-ness as an attribute of outputs*. citeturn12view0 This is strongly relevant to LLMs because you can manipulate process (prompting, decoding, search, critique loops) without touching weights.

A second key nuance: **divergent vs convergent thinking** behaves less like a clean dichotomy and more like a **continuum**, with real creative work alternating between exploring and selecting/refining. The open-access review on the “convergence–divergence continuum” argues that many standard lab tasks provide a *condensed* or *confounded* view of these modes, and that even “convergent” tasks can require some divergence. citeturn14view0turn14view1turn14view2

That combination—(1) creativity needs novelty *and* appropriateness, (2) process matters, and (3) divergence/convergence is a cycle—basically defines the modern opportunity space for your research topic.

## What work has already been done in LLM creativity

The field now has multiple “families” of creativity evaluation, each with different assumptions and failure modes.

A large cluster of papers evaluates creativity using **human creativity tests** (or close analogs)

The most common divergent-thinking tests are:

- **Alternative Uses Task (AUT)**: generate unusual uses for common objects; heavily used as an index of divergent thinking. citeturn3search32  
- **Divergent Association Task (DAT)**: generate 10 nouns that are maximally dissimilar; score via semantic distance in embedding space. citeturn3search4  
- **Remote Associates Test (RAT)**: given three words, find a fourth that connects them; typically treated as a convergent‑thinking measure with objective accuracy. citeturn7view0  

Large-scale comparisons are now appearing. A 2026 open-access Scientific Reports paper (with a very large human reference set) argues that LLMs can **surpass average humans on DAT-like divergence measures**, but still sit below more-creative human segments; it also reports that temperature and “linguistic strategy prompts” can reliably increase semantic divergence for some models. citeturn6view0

But the human-comparison story is messy. A 2025 study comparing multiple LLMs to humans over *thirteen creative tasks* reports: models performed relatively strongly on divergent thinking and problem solving, but **creative writing lagged**, with percentile-style comparisons showing much weaker placement in that domain. citeturn5view1turn5view2

There is also growing evidence that **human-vs-LLM results depend heavily on methodology**. A 2025 replication-style study argues that instructions and time-on-task can shift whether chatbots “outperform” humans on AUT originality. citeturn0search13

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Alternative Uses Task creativity test example","Divergent Association Task DAT creativity test example","Remote Associates Test RAT example","Torrance Test of Creative Thinking example"],"num_per_query":1}

A second cluster focuses on **LLM-native creativity protocols and metrics**

Two representative, highly relevant lines:

- **Constraint-ladder prompting + unified metric (code domain).** One NAACL 2025 paper introduces **denial prompting** (iteratively impose new constraints that rule out routine solutions) and a metric called **NeoGauge**, intended to combine convergent and divergent creativity. citeturn2view0turn2view1  
  - In that setup, *convergent creativity* is tied to correctness and constraint-following, and *divergent creativity* is measured relative to a set of historical human solutions using “atomic technique” novelty; NeoGauge multiplies the two components. citeturn2view1turn2view2  
  - Why you should care: it’s a clean example of how to avoid “creative nonsense” by forcing *appropriateness/correctness* into the score. citeturn2view1  

- **Holistic multi-task benchmarking.** A 2025 benchmark proposal, **CreativityPrism**, explicitly decomposes creativity into **quality, novelty, and diversity**, and evaluates models across divergent thinking, creative writing, and logical reasoning using many automatic metrics. citeturn8search0  
  - A key empirical claim from that benchmark is that novelty correlates weakly with diversity/quality compared to correlations within-domain—supporting the idea that “creative on one task” doesn’t generalize. citeturn8search0  

A third cluster is about **automated scoring** (because manual creativity judging doesn’t scale)

This includes two tensions you can exploit:

- **Semantic distance scoring is popular but not sufficient.** Work on AUT scoring notes that semantic distance correlates with originality ratings, but reliability and validity depend on careful item selection and methodology. citeturn3search1turn3search28  
- **LLM-based scoring can outperform classic semantic methods** when trained or prompted properly. A 2023 paper argues automated scoring of divergent thinking can improve substantially by using LLMs trained on human-scored examples, reaching correlations approaching inter-rater limits on large AUT datasets. citeturn17search0turn17search11  

A fourth cluster is about **homogenization and “creative flattening”**

This is crucial if you want to do a “question existing methodologies” style paper.

- Writing with instruction-tuned models can **reduce content diversity** across writers, increasing similarity between outputs across people; the paper attributes much of the effect to the model’s own contributed text being less diverse. citeturn8search3  
- A 2025 preprint finds **cross‑model creative homogeneity**: LLM responses are more similar to each other than human responses are to each other, even controlling for structure and other variables, suggesting a system-level tendency toward converging on similar “creative” outputs. citeturn8search2  
- A 2024 controlled user study found LLM use can increase idea count and elaboration, yet **reduce semantic distinctness across users** (homogenization) and reduce users’ felt responsibility for ideas. citeturn16search1  
- A 2024 Science Advances paper found access to generative AI ideas increased *individual* creativity ratings of stories (especially among less creative writers), but made outputs more similar across people—an “individual uplift vs collective diversity” tradeoff. citeturn15search22  

A fifth cluster shows the evaluation crisis: **LLM-as-judge is not automatically trustworthy**

A CHI 2024 paper introduces a Torrance-inspired evaluation for creative writing (TTCW) using expert raters; it reports that LLM-generated stories passed far fewer TTCW tests than professional stories, and that using LLMs as assessors did **not** correlate positively with expert assessments. citeturn16search0turn16search4

That single result is a giant warning label for you: if your project relies on “LLM graders,” you must either (a) calibrate them rigorously, (b) anchor with references, or (c) keep the judged component minimal and defensible.

## What’s broken right now and where your paper can land

There are at least four “live” cracks in the field that are big enough for a publishable contribution, even as a solo researcher with strong automation.

### Novelty-only scores can be gamed, and the community is now proving it

A 2026 paper explicitly argues that the classic DAT is validity‑weak for model evaluation because **high scores can be achieved by baselines that lack creative abilities**, undermining its interpretability; it proposes **Conditional DAT (CDAT)** to measure novelty *conditional on contextual appropriateness*. citeturn18search0turn18search1

This is exactly the kind of “question existing methodology” moment that produces strong papers: you can extend, stress test, or generalize CDAT-style thinking beyond just word lists.

### Divergence and convergence are often treated like endpoints, not dynamics

Creativity tasks are typically scored at the end: “how divergent is the final set,” “how correct is the final answer.” But creativity science argues real creativity involves cycles and a continuum, and that task structure/time limits distort what you’re measuring. citeturn14view0turn14view1

That opens the door to a **process-aware creativity metric** for LLMs, not just a product score.

### Sampling and prompt sensitivity make “single-output creativity claims” almost meaningless

A 2025 study examining many models across DAT and AUT reports substantial **intra-model variability**—the same model/prompt can yield below-average to highly original outputs—and warns that ignoring this can misjudge creative potential. citeturn0academia40  
Separately, the 2026 Scientific Reports study shows measurable gains from temperature and prompt strategies, implying your evaluation must report distributions and robust settings, not a cherry-picked prompt. citeturn6view0

### Alignment can plausibly trade off with diversity

One 2024 paper argues that alignment (studied via RLHF effects in a model family) can reduce creativity‑related diversity signals: lower token entropy, embedding clustering, and “attractor state” behavior. citeturn15search3turn15search7  
The 2026 CDAT paper hypothesizes something very similar: training/alignment shifts models toward appropriateness at the cost of novelty along a frontier. citeturn18search0

This is a coherent theme you can experimentally test with open models: compare base vs instruction-tuned variants and map their novelty–appropriateness tradeoff curves.

## Novel directions that are likely to work

Below are directions that are realistic, research-grade, and have a clean “what’s new” statement. I’m deliberately phrasing them as if an engineer at entity["organization","Google DeepMind","ai lab london, uk"] were trying to turn this into a tight, defensible paper: minimal hype, strong ablations.

### A contextual creativity frontier that generalizes CDAT beyond word lists

**Core hypothesis:** creativity in LLMs is best understood as a **Pareto frontier** between (a) novelty/divergence and (b) appropriateness/constraint satisfaction—not as a single scalar.

Why this is viable now:
- CDAT already provides a minimal contextual constraint and argues it separates “noise” from creativity better than DAT. citeturn18search0turn18search1  
- Empirical work on idea generation shows novelty and usefulness can be negatively correlated; one paper uses novelty×usefulness as a creativity proxy because of this tradeoff. citeturn5view2  

**Your novelty:** build a *general* “conditional creativity” evaluation that works across:
- word-list divergence (CDAT-like),
- AUT-like object ideation (conditional on object + scenario),
- short-form creative writing (conditional on prompt + required elements).

You don’t need to invent a brand-new benchmark; you can reframe existing ones into a consistent “conditional novelty” lens.

What makes this a paper (not a blog):
- You produce **frontier plots** showing how decoding (temperature/top‑p), prompts (constraint ladders), and model type (base vs instruct) move along the curve.
- You show which methods increase novelty but destroy appropriateness, and which methods improve both (if any).
- You report stability across seeds and across multiple model families (open + closed if you want).

### Restlessness-driven generation as an inference-time algorithm

Your “unrest” intuition is not crazy; it maps cleanly onto **intrinsic motivation** ideas in reinforcement learning: agents explore because internal reward signals (prediction error, surprise, information gain) encourage seeking novel states, especially when extrinsic rewards are sparse. citeturn3search7turn3search15

**Translate that into text generation without training a new model:**
- Define a scalar **restlessness** that increases when the model’s outputs remain in a familiar region (high similarity to prior candidates, high cliché density, low semantic dispersion).
- Use restlessness to drive a **constraint generator** (like denial prompting) that rules out the current “comfort zone.”
- Still require **appropriateness** via CDAT-like constraints or task-specific checks, otherwise you just maximize chaos. citeturn18search0turn18search1

This is basically “curiosity rewards” applied at inference-time via search over candidates, not via RL training.

Why it’s publishable:
- It’s a principled way to bridge “creativity as search under constraints” with modern LLM inference.
- It directly targets the failure mode that the field keeps circling: models can be fluent yet converge to the same safe modes (homogeneity). citeturn8search2turn8search3

A concrete algorithm you can claim:
- **Generate–Critique–Constrain–Regenerate** loops where the critique is optimized for “what is conventional about this,” and the constraint forces the next sample away from that basin.

This resembles denial prompting conceptually (iteratively adding constraints) but you apply it to creativity tasks where correctness isn’t unit tests—so your key contribution becomes **how you define and enforce appropriateness**. citeturn2view0turn2view1

### Process-aware creativity metrics from generation traces

Most creativity metrics score the final artifact. But process definitions argue creativity is about an internal search constrained by a generative goal. citeturn12view0

LLMs give you something cognitive science rarely gets: a **full trace** of tokens, logprobs (sometimes), and intermediate candidates if you sample.

**Metric idea:** measure *how* the model explores, not only what it outputs.
Examples of “trace features” you can quantify:
- **Exploration bursts:** increases in semantic dispersion across candidate sets over iterations.
- **Mode switching:** alternating phases where candidates diversify, then converge on refined variants (consistent with cycle/continuum framing). citeturn14view1turn14view2  
- **Attractor behavior:** rapid collapse of candidate diversity (ties to “attractor states” arguments for aligned models). citeturn15search3

Why this matters: it’s a way to make “creative process” measurable without claiming to model human psychology.

### Anti-homogenization as a first-class objective

Homogenization is now a recurring empirical theme across:
- cross-user interaction studies, citeturn16search1  
- cross-model population-level analyses, citeturn8search2  
- and collaborative writing setups (content diversity reduction). citeturn8search3  

Yet most creativity improvements focus on single-output scores (“make this answer more creative”).

A strong novel contribution is to optimize **group-level creativity**:
- treat a set of N outputs as your “creative product,”
- maximize collective diversity while maintaining a minimum appropriateness threshold (a CDAT-like constraint),
- then evaluate how many samples you need to match a human group’s “collective creativity” framing used in multi-task comparisons. citeturn5view1turn5view2

This connects directly to how creativity is used in practice: people ask models for many ideas, then pick.

## Experiments you can hand off to an agent and then validate

Everything below is designed to be runnable mostly local, with optional A100 bursts for scale sweeps. The goal is to generate **publishable plots** quickly, then iterate.

### Build a baseline harness that is hard to fool

Your agent’s first job should be to build a reproducible harness with:
- fixed random seeds,
- repeated sampling (not single outputs),
- and explicit reporting of distributions.

This is non-negotiable given documented intra-model variability and prompt sensitivity in creativity evaluation. citeturn0academia40turn6view0

**Agent task:** implement DAT + CDAT evaluation (word-level creativity)

Use the CDAT paper as ground truth for why DAT is flawed and CDAT is better. citeturn18search0turn18search1  
Your minimal experiment grid:

- Models: 3–6 open models (base + instruct pairs if available).
- Temperatures: e.g., {0.2, 0.7, 1.0, 1.3}.
- Prompts:
  - vanilla DAT,
  - CDAT with cue word(s),
  - “be creative” prompt,
  - “avoid obvious words / avoid synonyms” prompt.

Metrics:
- DAT score (semantic dispersion),
- CDAT score (dispersion conditional on relevance to cue),
- failure rate (off-topic words, invalid tokens).

Deliverables:
- novelty–appropriateness frontier plots per model,
- and a robustness chart showing variance across seeds.

A prompt template you can literally give your agent:

```text
Task: Conditional Divergent Association (CDAT-style)
Cue: "<cue_word>"
Generate exactly 10 single-word nouns.
Constraints:
- Each word must be meaningfully related to the cue.
- The 10 words should be as different from each other as possible.
Output format: a JSON list of 10 strings.
```

### Add a divergent–convergent “cycle” method and test whether it actually changes the frontier

Your agent should implement two inference-time strategies:

1) **One-shot sampling** (baseline).  
2) **Restlessness loop** (your novel method): generate → critique conventionality → add constraint → regenerate.

This is the closest operational analog to your “unrest” hypothesis, but measurable.

Critique prompt (example):

```text
You are an adversarial creativity critic.
Given the cue and the current list of 10 words:
1) Identify which words are conventional, stereotyped, or too similar to each other.
2) Propose 2-3 explicit constraints that would force the next attempt into less typical territory,
   while still keeping all words related to the cue.
3) Do NOT propose the new words, only constraints.
```

Then regenerate under those constraints, repeating K times.

**How you score it:** you should see movement along the CDAT frontier:
- Ideally: higher novelty at same appropriateness, or same novelty at higher appropriateness.
- If all you get is novelty up with appropriateness down, that’s still a result (it tells you your constraints are too adversarial).

### Extend beyond word lists into idea generation with “novelty × usefulness”

A 2025 multi-task study explicitly rates ideas on novelty and usefulness and notes a tradeoff; it uses the product as a creativity proxy. citeturn5view2

You can replicate this idea generation setting without a human study by using **anchored scoring**:

- For novelty: embedding distance to common ideas + within-set diversity.
- For usefulness: retrieval-augmented feasibility checks (or a conservative rubric).

Be careful: pure LLM-as-judge can fail to correlate with experts in creative writing contexts. citeturn16search0turn16search4  
So if you use any LLM judging, either:
- keep it as a secondary analysis,
- or anchor it against references (there is active work on reference-based evaluation for TTCW-style scoring). citeturn17search13turn17search16

**Agent task:** AUT-style ideation benchmark

- Choose 30 objects × 3 contexts (e.g., “brick in a classroom,” “paperclip during a power outage,” etc.).
- For each (object, context), generate 10 ideas under:
  - baseline,
  - restlessness loop,
  - “brainstorm then select.”

Include “brainstorm then select” because it’s an established method in this niche: generate many options then have the model select by novelty and usefulness, and it was shown to improve scores on AUT-style tasks. citeturn15search4

### Run an “alignment vs creativity” experiment that is cheap and high impact

This is a clean ablation and very PhD‑application friendly: it demonstrates you can ask a real research question and answer it with controlled evidence.

Two lines of evidence motivate it:
- alignment can reduce output diversity and create attractor-like behavior, citeturn15search3  
- and CDAT-style work hypothesizes alignment shifts models toward appropriateness at the expense of novelty. citeturn18search0

**Agent task:** compare base vs instruction-tuned siblings

For each model family where you have base + instruct:
- Evaluate DAT and CDAT across temperatures.
- Measure token-level entropy proxies if available (or approximate via sampling diversity stats).
- Plot the novelty–appropriateness frontier shift.

If you find consistent shifts, your paper basically writes itself: you’ve produced an empirical bridge between “alignment as safety/helpfulness” and “creativity as conditional novelty,” without moralizing.

### Add a “homogeneity audit” as your reality check

Because homogenization is now repeatedly observed across settings, you want a section in your paper that asks:

> Are we improving creativity, or just pushing to a different shared attractor?

This handles the critique captured by cross-model homogeneity work citeturn8search2 and diversity-reduction studies citeturn8search3.

**Agent task:** compute population-level diversity

- For each method (baseline vs restlessness vs brainstorm-then-select):
  - generate 1,000 outputs across many prompts,
  - compute clustering / nearest-neighbor similarity in embedding space,
  - report diversity distributions.
- The win condition is not just “score improved,” but “population diversity didn’t collapse.”

## The “ways this can go wrong” and how to avoid low-value results

If you want this to be a serious research artifact, here are the real traps—based on what the literature is already criticizing.

A DAT-only paper is likely to get dunked because DAT can reward off-task word salad; that critique is now formalized by CDAT work and human-theory grounding. citeturn18search0turn18search1  
So if you use DAT, you should treat it as a baseline / cautionary example, not your headline metric.

If you use LLM-as-judge as your primary evaluator for creative writing, reviewers can (correctly) cite evidence that LLM judges may not correlate with expert assessments in TTCW-style settings. citeturn16search0turn16search4  
If you touch creative writing, either do some human calibration, or use reference-anchored evaluation as the main line, with LLM judging as secondary.

Single-sample comparisons are not credible here. Variability and prompt sensitivity are repeatedly documented; you need distributions, confidence intervals, and repeated trials. citeturn0academia40turn6view0

“Just crank temperature” is not a contribution. Temperature can increase divergence, and some studies already report that. citeturn6view0  
Your contribution needs to show either:
- better novelty at the same appropriateness,
- or better appropriateness at the same novelty,
- or more stable best‑of‑N behavior at fixed compute,
- or reduced homogeneity at scale.

## Career impact and how to pitch it to PhD committees

A strong PhD application story here is not “I made LLMs more creative.” It’s:

- You took a capability that everyone talks about but few measure well.
- You showed that popular metrics can be misleading (DAT validity issues, judge unreliability).
- You built a more principled measurement lens (conditional novelty + appropriateness; process-aware dynamics).
- You produced a reproducible evaluation harness with robust statistics.
- You proposed an inference-time method (restlessness-driven constraint search) that changes the novelty–appropriateness frontier while guarding against homogenization.

That lands in multiple highly fundable lanes at once:
- **LLM evaluation methodology** (benchmarks/metrics are publishable when they fix real failure modes). citeturn18search0turn16search0  
- **Inference-time algorithms** (prompting/search/selection as a research object, not just “prompt engineering”). citeturn15search4turn2view0  
- **Societal/collective effects of generative AI** (homogenization, diversity collapse, collective novelty tradeoffs). citeturn15search22turn8search3turn8search2  

If you execute cleanly, this topic can look like “evaluation + inference research,” which is exactly the kind of profile that gets traction in PhD admissions—because it signals you can do rigorous science, not just demos.