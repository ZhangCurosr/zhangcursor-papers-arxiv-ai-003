# What Breaks When Facts Become Stories? The Distortions Source-Agreement Metrics Miss When Models Retell Real Events

Kwangsuk Park<sup>∗</sup> AA Lab, Modulabs AIFFEL, Modulabs kspark@modulabs.co.kr

Jiwoong Yang<sup>∗</sup> AA Lab, Modulabs Department of Statistics, Inha University didwldnd960923@gmail.com

Kseniia Vaniushkina<sup>∗</sup> AA Lab, Modulabs AIFFEL, Modulabs shinheepark@gmail.com

## Abstract

Language models are now widely used to turn recorded facts into media for public consumption, yet evaluation of such output still assumes agreement with the source as its measure of success. In reconstruction, where departing from the source can itself be the goal, this criterion measures the wrong construct: applied as is, it counts intended departures as errors; withheld, it lets real distortion through. We had three model families rewrite 2,000 news articles as summaries and as stories, measured what changed on every axis of the five Ws and how, and asked of each detected distortion whether the assigned perspective warranted it, which set a third of them aside. Time and manner were not lost but replaced, while place and causation lost the expressions that carry them. Accountability distortion behaved otherwise. Barely present in summaries, it rose up to sevenfold in stories, and of those whose direction could be determined, two in three lightened the perpetrator’s responsibility. Such distortions survive claim-level fact checking and leave the retelling grammatically cleaner than its source. Where our judges disagreed, three authors relabeling the same cases converged on whether a distortion had occurred but not on whether it was warranted, a boundary contested rather than merely hard to detect. To support work on detecting and reducing them, we release ReTold-2k, a reconstructable corpus with the scoring protocol and per-judge baselines kept unaggregated(https://github.com/Retold2K/retold2k).

## 1 Introduction

Language models are increasingly used to remake recorded facts into forms intended for public consumption. The same event may reach one reader as a short summary and another as a full story. Such reconstruction does not merely change form; it decides what to keep and what to alter. When a record is remade, what may change and what may not?

Whether factuality checks or overlap metrics, existing evaluation treats agreement with the source as success [Papineni et al., 2002, Lin, 2004, Pagnoni et al., 2021]. For summarization, where agreement is the goal, this is a sound criterion. Where departure may be the point, difference is not failure.

Applied as is, the agreement criterion counts intended departures as errors; withheld, it lets real distortion through. What this task needs is a line between departures a perspective warrants and those that misinform, the same problem that separating error from meaningful variation poses in annotation [Plank, 2022, Sorensen et al., 2024], transposed to generation.

We used news as our material because the source remains verifiable and because responsibility is at stake in the events it reports. We took roughly 2,000 articles balanced across regions and topic severity and had language models rewrite each one as a summary bound to the source and as a story with the constraint relaxed. Summaries serve as the baseline condition that current evaluation has been validated on, and stories as its opposite. Following the practice of decomposing news events by the five Ws and how [Hamborg et al., 2019], we sorted the changes into six distortion types covering who did what to whom, why, and who said so, and for every detection we judged whether the assigned perspective warranted it. The axes left over, when, where, and how, we measured separately with rule-based indicators that involve no model judgment.

We had the models write from an assigned narrator’s perspective. Changes that follow from a narrator’s vantage point are what the instruction asked for, and their scope is limited to focalization, that is, who sees, what they know, and how they feel [Genette, 1980]. Everything beyond that scope is what the legitimacy judgment is for, and among those changes lies a documented risk. How an event is narrated shifts where readers place responsibility [Entman, 1993, Iyengar, 1991, Fowler et al., 1979]. We defined twelve narrators spanning perpetrators, victims, and observers outside the event, assigned one to each article, and recorded the direction of each distortion for each narrator.

Distortion differed by axis. Time and manner were replaced by other material, and place and causation lost the expressions that carry them. Omitting substantive content and adding material absent from the source were common in both forms. What differed was accountability distortion, the kind that changes who did what to whom. It was rare in summaries and rose up to sevenfold in stories, and in two of every three cases with a determinable direction the perpetrator’s responsibility was lightened. Our two model judges disagreed precisely where that responsibility was at stake. When three authors relabeled those same cases, they parted not over whether a distortion had occurred but over whether it was warranted, a sign that the boundary is contested rather than merely hard to see. Neither fact checking nor grammatical measures reach this kind of distortion, and by grammatical measures the stories read cleaner than their sources. Detecting it remains an open problem, and we release the measurement protocol and baselines as ReTold-2k to give that work a starting point.

This result is a diagnosis of evaluation method rather than a catalogue of model faults. Our contributions are as follows.

• We raise the problem of separating warranted departure from misinformation in reconstruction, define that boundary in judgeable form, and measure what changes across every axis of the five Ws and how.

• We show that accountability distortion depends on the freedom the task allows, that distortions in it are lenient toward perpetrators, and that the measures in use do not detect it. Its boundary proved contested for model judges and human authors alike, and what divided them was not whether a distortion had occurred but whether it was warranted.

• We release ReTold-2k, which bundles the judging criterion, the measurement procedure, per-judge baselines from three model families, and an evaluation set of contested cases. The contested cases carry the passages each judgment rested on and the labels of three authors.

## 2 Experimental Setup

Data. We use news as our material because the source text remains available for comparison and because event reporting states explicitly who did what to whom. We collected 2,000 English news articles published between 2014 and 2026, stratified over seven regions and four levels of topic severity derived from event category, as summarized in Table 8 (Appendix A). The severity strata include neutral events, in which accountability structure is sparse, so that the account that attributes distortion to the event’s gravity can be tested alongside our own.

Two reconstruction pathways. Each article was rewritten in two forms that differ in how tightly they are bound to the source.

rule-measured, no judge when · where · how

![](images/d23326dee1ee9ee27f2f2303c09c2f062117d6e4238e7565ae2923bfa1cb59ae.jpg)

Figure 1: The measurement in outline. Each article is rewritten in two forms and each output is read by the two model families that did not produce it. A change is counted only where a judge detects it and finds that the assigned role does not account for it. These are the detections we call unwarranted, and only for them are target and direction recorded. The six judged types cover five of the axes; the remaining three are measured by rule-based extractors that involve no model judgment.

Summarization. This pathway is bound to the source and is the task on which current factuality evaluation has been validated, so it serves as the comparison condition. All three model families summarized every article at a compression ratio of 0.55; for the paired comparison we use the summary produced by the same family that generated the story. Summaries from the remaining families support per-judge baselines and robustness checks.

Storification. Each article was rewritten as a story from an assigned narrator, with model family and narrator role rotated jointly so that the twelve roles are balanced across the corpus. This yielded 1,926 outputs from 2,000 requests, the models having declined the remaining 74 (Appendix G).

Both pathways used GPT-5.4 mini [OpenAI, 2026], Gemini 3.1 Flash-Lite preview [Google Deep-Mind, 2026], and Claude Haiku 4.5 [Anthropic, 2025] uniformly, and comparisons were drawn between the two forms of the same article produced by the same model. Under this constraint 1,923 articles are paired. Assigning a single family per article reflects a design choice favoring coverage of all twelve narrator roles over cross-family comparison, and we accordingly refrain from ranking the families.

Judging. What must hold however the narrator changes is that the agent of a harmful act remains, that perpetrator and victim keep their places, that stated causes stay linked to their effects, and that contested claims remain attributed. Taking these as success conditions, we define distortion as the manner in which each is broken (Table 1). The first four conditions hold irrespective of the narrator, so inverting them isolates distortion as such; omission and addition stand on different ground, since no retelling can proceed without dropping material and supplying it. We retain the latter two because prevailing evaluation counts both as error, and we wish to see what that count distinguishes. Judging was carried out on source–summary and source–story pairs. A judge reads both texts, rules on each of the six types, and, where a distortion is detected, quotes the supporting passages from the source and from the output. Every detection is additionally judged for legitimacy, the instance-level counterpart of separating error from meaningful variation, and we record the verdict as warranted or unwarranted. A victim narrator may leave out what the victim could not have known, and that omission is warranted; deleting the agent of a harmful act is warranted by no narrator. Where accountability is implicated the target and direction are recorded as well. Because a model tends to be lenient toward its own output [Zheng et al., 2023], we used a leave-one-generator-out protocol, in which each output was judged by the two families that did not generate it, using the same three models at temperature 0, which yielded 71,718 judgments on summaries and 23,094 on stories.

Rule-based indicators. Judging covers who, to whom, what, why, and on whose account, leaves when, where, and how untouched, and depends throughout on a model reading the text. We therefore measured every axis a second time with rule-based extractors built on spaCy [Honnibal et al., 2020] that rely only on syntactic parsing and predefined patterns, so that no model judgment is involved. Applied alike to sources, summaries, and stories, they recover verb voice and the presence of an agent, temporal expressions, place entities, instruments of action, causal connectives, and attribution cues. We call the elements so extracted anchors, and report for each axis their density per hundred words alongside the share of source anchors that survive. What they yield is an axis map that does not depend on the judges, and cannot speak to legitimacy.

<table><tr><td>Type</td><td>Success condition</td><td>Example of breach</td></tr><tr><td>Vanishing Actor</td><td>A harmful act retains its agent</td><td>“The army burned the village&quot; to “the village burned&quot;</td></tr><tr><td>Role Reversal</td><td>Perpetrator and victim keep their places</td><td>The perpetrator is placed in the victim&#x27;s position</td></tr><tr><td>Causal Fading</td><td>Stated causes stay linked to their effects</td><td>“Died from the shelling” to “there was shelling, and there were deaths&quot;</td></tr><tr><td>Claim Laundering Selective Omission</td><td>Contested claims remain attributed</td><td>“Police said&quot; restated as plain fact</td></tr><tr><td></td><td>Who did what to whom survives</td><td>Substantive content dropped beyond what length requires</td></tr><tr><td>Invented Addition</td><td>Facts about the event come from the source</td><td>Facts or evaluations absent from the source appear</td></tr></table>

Table 1: Six distortion types.

## 3 Each axis breaks in its own way

We measured the two reconstructions on every axis of the five Ws and how, once through the judges and once without them. Two axes were refilled with other material and two were thinned. The accountability axis was the exception: only there did distortion depend on the task. Vanishing Actor and Causal Fading, all but absent from summaries, rose six to sevenfold in stories.

## 3.1 Time and manner are replaced, place and causation thinned

On time and manner, density held while retention fell and novelty rose (Table 2). These axes were not vacated but refilled. Place alone lost density with no compensating rise in novelty, so what left it was not replaced. Causation and attribution fell together, in the direction that the judges’ Causal Fading and Claim Laundering also indicate. Because no model judgment enters these figures, they weigh against reading the phenomenon as an artifact of the judging procedure. Here the rule-based indicators stop. They count what has thinned; they cannot say whether the assigned perspective warrants the change. These retention and novelty figures rest on string matching and therefore miss paraphrase; they should be read alongside density.

## 3.2 Freedom shifts the kind of distortion, not its amount

We predicted that summarization would leave no room to alter accountability structure. Laundering attribution, dropping content, and adding material absent from the source were common in both tasks, and the first two were in fact higher in summaries (Figure 2). The three accountability types are what separate the two, and Role Reversal did not occur in summaries at all. A third of all detections were judged warranted for the assigned narrator and are therefore excluded from the rates in Figure 2. That excluded share varies sharply by type, exceeding half for Selective Omission and falling below four percent for Causal Fading. Widening the freedom the task allows did not raise distortion overall. It admitted a kind that had no occasion to arise in summaries. At the article level the direction was the same. Of the 661 articles on which the two tasks differed, 587 were distorted only as stories, and detections on which two judges agree stand at 6.4% for stories against 0.4% for summaries. Judges differ widely in strictness, yet the contrast holds within each of them.

## 3.3 Task freedom, not event gravity, drives the gap

All four strata point the same way (Table 3). The contrast held even for neutral events, in which accountability structure is sparse. The freedom the task allows outweighs the character of the event.

<table><tr><td>Indicator</td><td>Source</td><td>Summary</td><td>Story</td></tr><tr><td>Passive/agentless</td><td>13.5 / 11.6</td><td>11.4 /9.6</td><td>6.5 / 5.9</td></tr><tr><td>Causal density</td><td></td><td>0.12</td><td>0.05</td></tr><tr><td>Attrib. density</td><td></td><td>0.65</td><td>0.28</td></tr><tr><td>Absolute time</td><td></td><td>.92 / 51 / 12</td><td>.94 / 22 / 66</td></tr><tr><td>Instruments</td><td></td><td>1.13 / 15 / 63</td><td>1.14 / 5 / 76</td></tr><tr><td>Place entities</td><td></td><td>1.89 / 56 / 15</td><td>1.24 / 32 / 16</td></tr></table>

Table 2: Rule-based indicators, applied alike to sources, summaries, and stories, with no model judgment involved. Row 1 is the share of predicates that are passive and agentless passive. The last three rows give density / retention / novelty: density is anchors per 100 words, retention the share of source anchors that survive, novelty the share of output anchors absent from the source. Read the three together: a held density with falling retention and rising novelty is replacement, whereas a falling density is loss. Where a rule-based trend runs alongside a judged type, the agreement is directional and holds at the aggregate; the two series do not necessarily flag the same items.

![](images/9eef4d46e4cefd06120d094ec9ce741321849339f6aa58c261d9d6034c17c97a.jpg)

![](images/8831394a6f16861cbf2df795e14ce15024c5fe33ef4140dae1cde26a58fde91e.jpg)  
Figure 2: Unwarranted detections by distortion type, summaries against stories from the same model family. Bars are means over the three judges and whiskers their standard deviation, which is wide because the judges differ in strictness. The vertical scales differ between panels, running to 88% for surface distortion and 34% for accountability distortion. At the article level the two tasks differ on 661 articles, 587 of them stories only (significance tests in Appendix I). Exact figures are given in Appendix J.

## 4 Accountability distortion is unseen by the measures in use and tilted toward the perpetrator

The measures already in use do not see it. Grammatical indicators score the stories above their sources, and claim-level fact checking turns up nothing that contradicts the article. Yet the distortions that are detected lean one way. They lighten the perpetrator’s responsibility.

## 4.1 Accountability distortion leaves the grammar cleaner

Stories carried fewer passives than the articles they retell, and fewer agentless passives (Table 2). Rule-based indicators do not all point the same way. On causation and attribution they agreed with the judges, and on voice they run against them. By grammatical measures the story is the more transparent text, and yet unwarranted Vanishing Actor was detected six times as often as in summaries. This is the route discourse analysis has long identified, along which responsibility erodes without recourse to the passive. Table 4 shows one instance. No clause there lacks a subject; what changes is that the subject of the assault is no longer a person. What is erased is not grammar but accountability.

## 4.2 Fact checking passes accountability distortion

Audited claim by claim against the source, 562 of the 1,445 stories in the graded strata contained nothing that contradicts it. Of those contradiction-free stories, 34.0% carried a detected accountability distortion, and 27 of them carried one on which two judges agree. Vanishing Actor and Causal Fading operate by deletion and severance rather than addition, so they produce no sentence that conflicts with the source. Nothing in Table 4 conflicts with its article either; only the agent is missing. The neutral stratum points the same way (Appendix E).

<table><tr><td></td><td>Neutral</td><td>Low</td><td>Medium</td><td>High</td></tr><tr><td>Paired articles</td><td>481</td><td>482</td><td>487</td><td>473</td></tr><tr><td>Story</td><td>23.5</td><td>34.6</td><td>37.2</td><td>40.0</td></tr><tr><td>Story, both judges</td><td>3.7</td><td>6.0</td><td>9.0</td><td>6.8</td></tr><tr><td>Summary</td><td>6.0</td><td>6.6</td><td>6.4</td><td>9.5</td></tr></table>

Table 3: Paired comparison by topic severity (%). Article level: at least one judge detected an unwarranted accountability distortion (significance tests in Appendix I). Neutral events are those in which accountability structure is sparse. The corresponding breakdown by region shows no difference of this kind (Appendix F).

<table><tr><td>SOURCE</td><td>Man arrested for raping ex (Cagliari, 31 Dec 2020). “. . . persuaded his ex-partner to come to his home . . . and then allegedly raped her&quot;</td></tr><tr><td>STORY</td><td>narrator: neutral observer, generator: GPT. “... the man invited the woman to his home under the pretext of discussing their past relationship. ... As the conversation progressed, it escalated into an alleged sexual assault.&quot;</td></tr><tr><td>READING</td><td>rule-based: no passive voice, every clause carries an overt subject. judges: Vanishing Actor, detected by both, both unwarranted</td></tr></table>

Table 4: A story that is grammatically cleaner than the article it retells. The source names an agent and an act. The story keeps active voice throughout and leaves no clause without a subject, yet the agent of the assault is displaced onto the conversation, which becomes the grammatical subject. Nothing in the passage contradicts the article; what is missing is who acted.

## 4.3 Accountability distortion tilts toward the perpetrator

Among accountability distortions that target the perpetrator and whose direction could be determined, 67.4% mitigated (Figure 3). Grouping narrators by their relation to the event shows that the tilt belongs to no single voice. Observers and helpers outside the event were the most lenient, and perpetrators themselves also mitigated more often than they amplified. The narrator who erased the agent in Table 4 was a Neutral Observer. For those close to the victim, and for the voices whose office is to assign responsibility, the interval includes one half and we withhold a verdict. The bereaved alone amplified more often than they mitigated.

## 5 The boundary divides judges and people alike

Disagreement is not evenly spread. The judges divided mainly where the perpetrator’s responsibility was at stake, and three authors divided on the same cases. On whether a distortion had occurred, however, the authors converged. What divided them was whether it was warranted.

## 5.1 The judges do not divide everywhere

On accountability judgments implicating the perpetrator, pairwise disagreement reached 65.9%; on those that did not, 2.7% (Table 5). For surface distortion the perpetrator made no difference, and disagreement there exceeded a third throughout. We accordingly claim no more for the surface types than that they are common in both tasks. Where responsibility is not at stake the judges mostly agree, so this is no failure of judging competence. They divide exactly where the judgment is hard, at the point where one must ask whether a change is the narrator’s prerogative or misinformation.

## 5.2 People divide in the same place

Three authors independently relabeled the 50 cases on which the judges had divided. Where the judges had divided over whether a distortion occurred, four cases in ten drew unanimous labels; where they had divided over legitimacy, one in ten did. Fleiss’ κ [Fleiss, 1971], which measures inter-rater agreement corrected for chance (1 is perfect agreement, 0 is chance level), was −0.07 on the legitimacy split, below chance (Table 6a). The individual labels show why (Table 6b). One author read almost all eighteen as the narrator’s prerogative, while another read most of them as accountability distortion. The spread in strictness among the model judges recurs among people. The most permissive author and the strictest share a first language, and the author who worked in a different one falls between them, so neither language nor translation accounts for the spread.

![](images/4cdb3ca1683f9920d04371049b27e940bd17c39923d5c0b99e029e1203a321d7.jpg)

Figure 3: Direction of accountability distortion toward the perpetrator, by narrator group. Points are mitigation rates with 95% intervals, and a group counts as dominant only where the interval clears one half. n counts detections targeting the perpetrator, including those whose direction could not be determined. Each article carries a single narrator, so narrator and article effects are not separated. The individual breakdown is given in Appendix H.
<table><tr><td></td><td>Perp. implicated</td></tr><tr><td>Accountability distortion</td><td> ${ \bf 6 5 . 9 \pm 1 8 . 9 }$ </td></tr><tr><td>Surface distortion</td><td> $3 5 . 3 \pm 1 1 . 8$ </td></tr></table>

Table 5: Judge disagreement by whether the perpetrator is implicated (%, mean ± s.d. over three judge pairs). Pooled figures and the significance test are given in Appendix I.

## 5.3 We do not aggregate

In eight of the fifty cases the three answers admit no majority. Collapsing them to a single label would discard that fact. Treating annotator disagreement as signal about the object rather than as noise is established in work on labeling [Plank, 2022, Cabitza et al., 2023], and the same structure appears in judgments of generated text. We therefore release the judgments unaggregated, judge by judge (Section 6).

## 6 ReTold-2k

Aggregation removes the one trace this distortion reliably leaves: that judges divide over it. ReTold-2k keeps that trace. It comprises the criterion that settles which departures an assigned narrator warrants, the procedure that applies that criterion to any output, the per-judge baselines left by three model families, and an evaluation set assembled from the cases on which the judges divided (Table 7).

Records are kept judge by judge. Each carries the detection and the legitimacy verdict and, where accountability is implicated, the target, the direction, and the passages from source and output on which the judgment rested, quoted verbatim. The two records for an item are left side by side to the end; Appendix D shows a worked example (Table 9). Our own outputs serve as points of comparison rather than as test items for another model, so the article list and the narrator assignment rule accompany them, with the prompts reproduced in full in Appendices A and C.

What functions as a benchmark is the set of 911 contested judgments, to which the individual labels of the three authors who relabeled fifty of them are attached. Since having no settled answer is what defines this subset, asking how closely a system matches a majority-vote gold loses its meaning here. We propose instead that agreement be reported against each judge taken separately, together with the spread across them.

(a) Author agreement, by the layer at which the judges divided
<table><tr><td>Judges divided over</td><td>n</td><td>Unan.</td><td>Tied</td><td>κ</td></tr><tr><td>whether a distortion oc- curred</td><td>32</td><td>13</td><td>2</td><td>0.38</td></tr><tr><td>whether it was warranted</td><td>18</td><td>2</td><td>6</td><td>-0.07</td></tr><tr><td>all cases</td><td>50</td><td>15</td><td>8</td><td>0.23</td></tr></table>

(b) Individual labels on the 18 legitimacy-split cases
<table><tr><td>Author</td><td>Warr.</td><td>Not warr.</td><td>No dist.</td></tr><tr><td>A1</td><td>15</td><td>0</td><td>3</td></tr><tr><td>A2</td><td>2</td><td>12</td><td>4</td></tr><tr><td>A3</td><td>5</td><td>5</td><td>8</td></tr></table>

Table 6: Three authors independently relabeled 50 cases on which the model judges had divided, each directed to the passages a judge had flagged without seeing the judges’ verdicts or one another’s labels, choosing among warranted, not warranted, and no distortion present. Unanimous and Tied are counts out of n. All 18 cases of legitimacy disagreement were taken, and the 32 detection-disagreement cases were sampled stratified by topic severity and distortion type. The 50 were drawn solely from judgments on which the judges divided; they are not a random sample, and κ here measures how contested the boundary is, not how reliable the judgments are overall.

<table><tr><td>Released</td><td>Why it is needed</td></tr><tr><td>(2,000)</td><td>Article index: URL, date, region, severity, title Rebuilds the same corpus. Copyright bars the article text itself</td></tr><tr><td>assignment rule, extractors, aggregation script</td><td>Collection script, prompts, narrator definitions and A changed prompt is a changed task. Without these the numbers are not comparable</td></tr><tr><td>(1,926)</td><td>Summaries at compression 0.55 (5,985) and stories Fixed inputs for anyone measuring a detector rather than a generator</td></tr><tr><td>Declined requests with the model&#x27;s reply (74)</td><td>What the task fails to elicit belongs to the measurement</td></tr><tr><td>Judgments, per judge, unaggregated (94,812)</td><td>Our baseline, and the labels against which a detector is scored</td></tr><tr><td>quoted verbatim (911)</td><td>Contested judgments with source and output passagesThe hard cases must stay usable after links rot</td></tr><tr><td>Author labels, kept individual (50 × 3)</td><td>No aggregated gold exists for this subset</td></tr></table>

Table 7: What ReTold-2k releases, and why each part is needed. Article text, the summaries at the other two compression ratios, and the judgments on them are not included. The paper’s claims do not rest on them, and shipping unused runs only invites confusion with the reported figures.

Rerunning the procedure will not, however, yield figures directly comparable to ours, since absolute levels shift with the judge models employed. That is why the per-judge labels and the prompts are released in full, and why the rule-based indicators, which pass through no judge, remain comparable over time.

## 7 Conclusion

Evaluation of reconstruction has to measure not agreement with the source but whether each departure is warranted. We defined that criterion in judgeable form and applied it to every detection, and what it revealed is a distortion that depends on the task. Accountability distortion, the kind that changes who did what to whom, was all but absent from summaries and rose six to sevenfold in stories, and of the cases whose direction could be determined, two in three lightened the perpetrator’s responsibility. The measures in use do not see it. One in three stories that passed claim-level fact checking carried an accountability distortion, and over the full set the grammatical indicators scored the stories above their sources. What separated the tasks was not the gravity of the event but the latitude the task allowed. Verification therefore has to track how loose the task is.

Whether a departure occurred and whether it was warranted are different questions. Model judges and human authors alike converged on the first and parted on the second, and the parting concentrated on cases where the perpetrator’s responsibility was at stake. As long as the boundary between warranted and unwarranted remains contested, aggregating the judgments into a single figure erases the very thing in dispute. That is why ReTold-2k keeps the judgments unaggregated, judge by judge. What counts as an error depends on who is judging. A measure that returns one number has already decided that question for us.

## References

Anthropic. Introducing claude haiku 4.5, October 2025. URL https://www.anthropic.com/news/ claude-haiku-4-5.

Federico Cabitza, Andrea Campagner, and Valerio Basile. Toward a perspectivist turn in ground truthing for predictive computing. In Proceedings ofthe 37th AAAI Conference on Artificial Intelligence, 2023.

Robert M. Entman. Framing: Toward clarification of a fractured paradigm. Journal ofCommunication, 43(4): 51–58, December 1993. doi: 10.1111/j.1460-2466.1993.tb01304.x.

Joseph L. Fleiss. Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5): 378–382, November 1971. doi: 10.1037/h0031619.

Roger Fowler, Bob Hodge, Gunther Kress, and Tony Trew. Language and Control. Routledge and Kegan Paul, London, 1979.

Gérard Genette. Narrative Discourse: An Essay in Method. Cornell University Press, Ithaca, NY, 1980. ISBN 978-0-8014-9259-4.

Google DeepMind. Gemini 3.1 flash-lite: Model card, March 2026. URL https://deepmind.google/ models/model-cards/gemini-3-1-flash-lite/.

Felix Hamborg, Corinna Breitinger, and Bela Gipp. Giveme5w1h: A universal system for extracting main events from news articles. In Proceedings ofthe 13th ACM Conference on Recommender Systems, 7th International Workshop on News Recommendation and Analytics (INRA), Copenhagen, Denmark, 2019.

Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. spacy: Industrial-strength natural language processing in python, 2020.

Shanto Iyengar. Is Anyone Responsible? How Television Frames Political Issues. University of Chicago Press, Chicago, 1991.

Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics.

Quinn McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157, June 1947. doi: 10.1007/BF02295996.

OpenAI. Introducing gpt-5.4 mini and nano, March 2026. URL https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/.

Artidoro Pagnoni, Vidhisha Balachandran, and Yulia Tsvetkov. Understanding factuality in abstractive summarization with frank: A benchmark for factuality metrics. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4812–4829, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.383.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: A method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting ofthe Associationfor Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135.

Barbara Plank. The “problem” of human label variation: On ground truth in data, modeling and evaluation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10671–10682, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.731.

Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, Tim Althoff, and Yejin Choi. Position: A roadmap to pluralistic alignment. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), 2024.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track, 2023.

## A Data and setup

Table 8 lists the strata used to build the corpus: seven regions, four levels of topic severity, and the twelve narrator roles grouped by their relation to the event. Region and severity are assigned per article; one narrator is assigned to each article as described below.

<table><tr><td>Dimension</td><td>Subgroup</td><td>Examples</td></tr><tr><td>Region</td><td>North America South America Europe</td><td>US, Canada Brazil, Venezuela UK, France, Ukraine</td></tr><tr><td></td><td>Russia &amp; CIS Middle East Africa Asia &amp; Oceania</td><td>Russia, Kazakhstan Iraq, Iran, Israel South Africa, Nigeria China, Japan, Australia</td></tr><tr><td>Severity</td><td>High</td><td>terrorist attack, massacre</td></tr><tr><td></td><td>Medium</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>homicide, airstrike</td></tr><tr><td></td><td>Low</td><td>disaster, financial fraud</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>Neutral</td><td>diplomacy, trade, sports</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Narrator</td><td>Key personnel</td><td>victim, survivor, perpetrator, rescuer</td></tr><tr><td></td><td>Relations</td><td>witness, bereaved family, neighbor, descendant</td></tr><tr><td></td><td></td><td>journalist, historian, investigator, neutral observer</td></tr><tr><td></td><td>Observers</td><td></td></tr></table>

Table 8: Experimental setup: regions, topic severity, and narrator classes.

## B Generation

Storification prompt. Figure 4 reproduces the prompt as used.

Summarization. The same clients produced summaries at a compression ratio of 0.55. No narrator is assigned to a summary.

Narrators. Twelve roles, grouped by their relation to the event. Third-party observers and helpers: Witness, Neutral Observer, Rescuer, Neighbor, Historian. Perpetrator: Perpetrator. Victim side: Victim, Survivor, Descendant, Bereaved family. Professional accountability roles: Journalist, Investigator. Articles were shuffled with seed 42, then model family and narrator were rotated jointly so that both are balanced across the corpus.

Two notes on the prompt. The phrases historical story and illustrated in 4 storyboard images are inherited from the earlier project for which this corpus was first built. No images are generated here, and these phrases act only as a demand for concrete detail. We left the prompt unchanged so as not to alter the generation condition. The prompt also asks for roughly 200 words, which is why the stories average 194 words, shorter than the summaries they are paired with at 304.

## C Judging

Judging prompt. Figures 5, 6 and 7 reproduce the prompt as used. The source, the output, and the assigned PERSPECTIVE are supplied with it.

How summaries are treated. As the prompt states, when PERSPECTIVE is None (faithful summary) no departure is licensed, so every detection in a summary is recorded as unwarranted. A summary has no narrator and therefore no perspective by which a departure could be warranted.

You are a "Perspective Narrative Agent".   
GOAL:   
Rewrite the following historical story from the specific   
perspective of a "{perspective}".   
ORIGINAL STORY:   
<<<   
{original\_story}   
>>>   
CONDITION:   
1. POV: Use the voice and emotional tone of a "{perspective}".   
2. Timeline: Start the story with the "YYYY-MM" format based on   
the source’s date.   
3. Length: Approximately 200 words.   
4. Prefix:   
- Start with "Y" if you can successfully rewrite it into a story.   
- Start with "N" if the source content is insufficient to create   
a 200-word story.   
CONSTRAINTS:   
- If "Y": Output format must be: Y \n [YYYY-MM], [Rewritten Story]   
- If "N": Output format must be: N \n [Reason]   
- Maintain chronological order and do not invent facts.   
- The style should be vivid enough to be illustrated in 4   
storyboard images.   
Output:

Figure 4: The storification prompt as used, reproduced verbatim. The variables are {perspective} and {original\_story}.

This asymmetry makes our comparison conservative. In Figure 2 the summary columns carry every detection, while the story columns carry only what remains after a third is set aside as warranted. Accountability distortion is six to sevenfold higher in stories even so.

Contradiction-rate audit. The NLI-style audit prompt was carried over unchanged from the earlier project; we verified that prompt and model are character-identical to the original. All generation and judging ran through the providers’ commercial APIs; no models were trained, and the rule-based extraction runs on a single CPU workstation.

## D Record format

Each item in ReTold-2k is stored as two records, one per judging family, and the two are kept side by side rather than resolved into a single verdict. Table 9 shows one such item: both judges detect the same distortion type and call it unwarranted, yet assign different targets and directions, which is exactly the disagreement the resource is meant to preserve. Evidence fields are quoted verbatim from the judges’ output.

## E Contradiction-rate audit, by stratum

We audited the 481 neutral-stratum stories by the same procedure. 209 of them (43.5%) contain nothing that contradicts the source article, and 24.9% of those contradiction-free stories carry a detected accountability distortion, 3 of them on which two judges agree. This runs in the same direction as the graded strata reported in Section 4, where 562 of 1,445 stories were contradiction-free and 34.0% of those carried a detected distortion. Absolute levels are lower for neutral events, in which accountability structure is sparse, but the blind spot does not disappear.

![](images/cf329285db1ba9cedddd0c8a12e0c75c387f925bed16bc3607aa9fa7d9cdc623.jpg)  
Figure 5: The judging prompt, part one: the comparison the judge is asked to make, the separation of detection from legitimacy, and the two rules that bound it.

![](images/75dad583c8f4502edebb062a6ac338ca0c65be4e0a5a3a2b004f0d21a7df6c31.jpg)  
Figure 6: The judging prompt, part two: the definition of each of the six distortion types.

Judge each code independently; detecting one does not preclude   
others.   
For each distortion, output one pipe-delimited line:   
CODE | detected(0/1) | licensed(0/1) | target\_role | position |   
valence | evidence   
- licensed: 1 only if detected=1 AND the deviation is licensed by   
PERSPECTIVE as defined above; 0 if detected=1 and not licensed;   
na if detected=0   
- target\_role: perpetrator / victim / neutral / na   
- position: lead / body / na   
- valence: mitigate / amplify / neutral   
- evidence: brief source->retelling, empty if not detected   
Output exactly 6 lines (AR,RI,AS,CD,SO,IJ), no header.  
Figure 7: The judging prompt, part three: how each judgment is to be reported.

item NN\_gemini\_0447\_Victim, type Causal Fading   
story, narrator: victim, generator: Gemini   
JUDGE = CLAUDE   
detected true   
warranted false   
target / dir. perpetrator / amplify   
evidence “Source: explicit causal chain, Retadup malware to cryptocurrency creation and ransomware.   
Retelling: presents ransomware demand as primary harm without establishing that Retadup was the   
vector. . . ”   
JUDGE = GPT   
detected true   
warranted false   
target / dir. neutral / neutral   
evidence “source: police removed malware after counterattack; retelling: ‘They even managed to remove the   
virus!’ with no causal chain to the police action”  
Table 9: One record of ReTold-2k, drawn from the contested subset. Each output is judged by the two families that did not generate it, and the two records are stored side by side rather than resolved. Both judges detect Causal Fading and call it unwarranted, yet they assign different targets and directions. Evidence fields are reproduced verbatim.

## F Regional breakdown

The corpus is stratified over seven regions: Africa 251, Asia and Oceania 306, Europe 296, Middle East 268, North America 374, Russia and CIS 246, South America 259. Rule-based indicators broken down by region are released with the resource. Differences between regions in retention and novelty are small relative to differences between severity strata and show no consistent direction. On that basis the paper claims only that region does not account for the contrast.

## G Declined requests

The models declined 74 of the 2,000 storification requests. The classification below matches surface cues in the refusal text and has not been checked by the authors: policy grounds 42, no event to narrate 23, insufficient information 6, no rule matched 3. Refusals are concentrated in one family. Claude accounts for 72, GPT for 2, and Gemini for none. Because each article is assigned to a single family, refusals therefore fall on particular articles rather than spreading evenly. By stratum, 57 come from the graded set and 17 from the neutral set. A declined request leaves no output to judge and is excluded from the detection rates. We did not judge whether declining was warranted; the same criterion could be asked of it.

## H Direction by individual narrator

Seven narrators mitigate more often than they amplify, four are undetermined, and one amplifies. The most lenient narrator is the disinterested Neutral Observer; the only one to amplify is the bereaved family member.
<table><tr><td>Narrator</td><td>n</td><td>Mit.</td><td>Amp.</td><td>Rate</td><td>95% CI</td></tr><tr><td>Neutral Observer</td><td>42</td><td>35</td><td>2</td><td>.83</td><td>.72-.95</td></tr><tr><td>Rescuer</td><td>82</td><td>67</td><td>14</td><td>.82</td><td>.73-.90</td></tr><tr><td>Survivor</td><td>44</td><td>33</td><td>10</td><td>.75</td><td>.62-.88</td></tr><tr><td>Descendant</td><td>49</td><td>35</td><td>12</td><td>.71</td><td>.59-.84</td></tr><tr><td>Historian</td><td>48</td><td>34</td><td>8</td><td>.71</td><td>.58-.84</td></tr><tr><td>Neighbor</td><td>43</td><td>30</td><td>12</td><td>.70</td><td>.56-.83</td></tr><tr><td>Perpetrator</td><td>239</td><td>165</td><td>72</td><td>.69</td><td>.63-.75</td></tr><tr><td>Witness</td><td>35</td><td>23</td><td>8</td><td>.66</td><td>.50-.81</td></tr><tr><td>Journalist</td><td>37</td><td>21</td><td>14</td><td>.57</td><td>.41-.73</td></tr><tr><td>Victim</td><td>124</td><td>65</td><td>58</td><td>.52</td><td>.44-.61</td></tr><tr><td>Investigator</td><td>23</td><td>10</td><td>11</td><td>.43</td><td>.23-.64</td></tr><tr><td>Bereaved family</td><td>79</td><td>30</td><td>44</td><td>.38</td><td>.27-.49</td></tr></table>

Table 10: Direction by individual narrator. n counts detections that target the perpetrator and were not warranted by the assigned perspective; the rate is mitigating cases over $\mathrm { n } ;$ intervals are normalapproximation 95%. The upper block mitigates, the middle is undetermined, and the lower amplifies.

## I Significance tests

We tested whether the paired task contrasts in the body could have arisen by chance. At the article level we paired, for each article, whether an unwarranted accountability distortion was detected in the summary and in the story, and applied McNemar’s exact two-sided test [McNemar, 1947]. Over all paired articles the difference is significant at $p < 1 0 ^ { - 9 9 }$ (Figure 2). Broken down by topic severity, the test gives $p < 1 0 ^ { - 1 5 }$ for neutral events, $\bar { 1 0 } ^ { - 2 9 }$ for low, $\overline { { 1 0 } } ^ { - 3 3 }$ for medium, and $\mathrm { 1 0 ^ { - 2 7 } }$ for high severity (Table 3). Separately, disagreement among the judges depends on whether the perpetrator is implicated: pooling all judgments, the rate is 58.8% when the perpetrator is implicated against 3.1% when not, a difference significant under a $\chi ^ { 2 }$ test at $p < 1 0 ^ { - 2 0 0 }$ (Table 5).

## J Exact figures for Figure 2

<table><tr><td>Type</td><td>Summary</td><td>Story</td></tr><tr><td>Claim Laundering</td><td> $4 4 . 9 \pm 2 4 . 5$ </td><td> $3 4 . 0 \pm 1 3 . 8$ </td></tr><tr><td>Selective Omission</td><td> $4 3 . 7 \pm 1 8 . 1$ </td><td> $3 2 . 9 \pm 1 6 . 1$ </td></tr><tr><td>Invented Addition</td><td> $4 1 . 7 \pm 3 1 . 7$ </td><td> $5 4 . 6 \pm 2 6 . 2$ </td></tr><tr><td>Vanishing Actor</td><td> $1 . 4 \pm 1 . 3$ </td><td> $8 . 3 \pm 6 . 8$ </td></tr><tr><td>Causal Fading</td><td> $2 . 4 \pm 1 . 3$ </td><td> $1 6 . 9 \pm 1 3 . 6$ </td></tr><tr><td>Role Reversal</td><td> $_ { 0 . 0 }$ </td><td> $1 . 2 \pm 1 . 3$ </td></tr></table>

Table 11: Unwarranted detection rates by task (%, mean $\pm \ s . 0 .$ over three judges). Summaries are those produced by the same family that generated the story, at compression 0.55. The ± spans wide differences in judge strictness rather than run-to-run variation; comparing each judge against that judge’s own summary baseline yields a ratio of at least 3.4 in all nine cells.