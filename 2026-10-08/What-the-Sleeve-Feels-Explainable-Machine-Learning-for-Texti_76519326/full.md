# What the Sleeve Feels: Explainable Machine Learning for Textile Pressure-Based Postural Screening

Limon Bin Hossain<sup>∗†</sup> and Md Sadib Rahman Ananta<sup>∗</sup>

<sup>∗</sup>Department of Industrial and Production Engineering,

Bangladesh University of Engineering and Technology (BUET), Dhaka, 1000, Bangladesh

<sup>†</sup>Corresponding author: mdlimonbinhossain@gmail.com

Abstract—Pressure-sensing smart textiles convert body-surface contact into a dense, image-like signal closely tied to posture and movement, making them a promising low-cost route to wearable posture screening. Realizing that promise, however, requires more than classification accuracy: a deployable system must generalize to wearers unseen during training, expose the physical evidence behind its decisions, and tolerate the small donning offsets that occur whenever a garment is removed and re-worn. This paper addresses these three requirements jointly using a knitted piezoresistive sleeve worn on the forearm as a testbed. We regroup fine-grained everyday activities into three coarser screening categories (neutral, potentially undesirable, and functional or transitional), engineer 29 interpretable pressuredistribution features spanning global intensity, spatial center of pressure, quadrant asymmetry, distribution complexity, and short-horizon temporal change, and evaluate under a strict subject-wise split. A tuned XGBoost classifier reaches 0.818 accuracy, 0.788 balanced accuracy, and 0.801 macro F1 on unseen test subjects, with tight frame-level bootstrap 95% intervals of about ±0.01 and a subject-to-subject standard deviation near 0.06 under leave-one-subject-out cross-validation. A simple 2D-CNN baseline trained on raw frames achieves broadly similar performance, showing that hand-engineered features are not left behind by a learned spatial representation on this task. SHAPbased explanation, a feature-group ablation, per-activity error analysis inside the pooled undesirable class, class-mapping sensitivity, and a simulated donning-rotation stress test together locate what the model relies on, where it degrades, and why, directly targeting the generalization, interpretability, and robustness gaps that determine whether such a system is deployable.

Index Terms—smart textile, pressure sensing, human activity recognition, explainable artificial intelligence, SHAP, wearable sensors, gradient boosting

## I. INTRODUCTION

Human activity leaves a continuous imprint on the pressure distribution between the body and its surroundings, whether through contact with a desk, a wall, a chair, or the body itself during folded-arm or self-touch behaviors. Pressure-mapping smart textiles convert this contact into a structured, imagelike signal that can be worn, washed, and integrated into clothing, offering a low-cost and unobtrusive alternative to camera-based or inertial motion capture for everyday posture recognition [1]–[3]. Piezoresistive and capacitive textile structures measure strain and contact pressure directly within the fabric rather than through rigid attached sensors [4], and matrix-style textile arrays have already been demonstrated for seated posture classification in a chair backrest and seat [5], for calibrated office-chair posture analysis [6], for eatingactivity monitoring with a capacitive neckband [7], [8], and for counting gym exercises with a resistive pressure mat [9]. These studies confirm that textile pressure matrices carry sufficient spatial information for fine-grained recognition, but nearly all of them instrument a fixed surface, a mat, or a neckband rather than a limb-worn sleeve, and few report performance on wearers unseen during training.

Human activity recognition from body-worn or ambient sensors is a mature area, with comprehensive surveys covering both feature-based and deep-learning pipelines [10]– [14]. Within this literature, posture-specific work has applied classical machine learning to pressure and force sensor data for sitting-posture classification [15] and has produced pressuremap datasets intended for general posture and subject analytics [16]. Across this body of work, three requirements that matter for deployment receive limited joint attention: generalization to wearers not seen during training, interpretability of which pressure characteristics drive a given decision, and robustness to the small donning offsets that occur whenever a garment is removed and re-worn. Interpretability matters because a system meant to support ergonomic or rehabilitation feedback cannot act as an opaque classifier, and robustness matters because a model whose decisions rest almost entirely on one spatial axis is fragile in exactly the direction a wearer is most likely to shift the garment.

Ensemble tree methods, particularly random forests [17] and gradient-boosted trees [18], remain strong baselines for tabular sensor features and are the primary and comparison models adopted here. Post-hoc explanation of such models is commonly performed with SHAP (SHapley Additive exPlanations), which attributes a prediction to individual input features through a game-theoretic Shapley value formulation [19] and admits an efficient exact implementation for tree ensembles [20], consistent with broader taxonomies of explainable artificial intelligence for building trust in automated decisions [21], [22].

This paper targets that combined gap: can pressuredistribution features derived from a knitted, matrix-based textile sensor array separate everyday postures into coarse screening categories under a subject-independent protocol, and can the resulting model be probed for the physical evidence behind its decisions, the sources of its errors, and its degradation under realistic donning variation? The sensing platform: a double-sided weft-knitted sleeve whose orthogonally stacked stripes form a 200-point piezoresistive matrix (40 $\mathrm { c m ~ \times ~ } 1 8$ cm) read out at 12-bit resolution and 50 Hz over Bluetooth. One stripe direction runs along the forearm and the other wraps around its circumference (Fig. 1 illustrates the resulting pressure signatures across everyday postures), so the two sensing axes separately encode proximal-distal and angular position, a distinction that becomes central to the explanation results.

![](images/0408aa6dd19387112cd5c8b014e330b366479a787fbfb8ae62940106d2ed6ac5.jpg)  
Fig. 1. Representative photographs and corresponding pressure maps for the 18 everyday activities in the Smart-Sleeve dataset. Image reproduced from the Smart-Sleeve dataset repository [23].

The contributions of this paper are as follows.

• We propose a deterministic, documented mapping from fine-grained everyday activities to three posture-screening categories and engineer 29 interpretable pressuredistribution features, spanning global intensity, activation area, spatial center of pressure, quadrant asymmetry, distribution complexity, and short-horizon temporal change, with all data-driven thresholds estimated from training subjects only.

• We combine SHAP-based explanation with a featuregroup ablation, a per-activity error analysis, and a classmapping sensitivity study to show which physical pressure characteristics drive each decision and how sensitive the reported performance is to labeling choices.

• We run a simulated donning-rotation stress test that circularly shifts sensor frames around the limb’s circumference, linking the resulting asymmetric degradation directly to the axis identified as most important by the explanation analysis.

## II. METHODS

## A. Dataset

The Smart-Sleeve dataset contains recordings from 14 healthy, right-handed subjects (3 female) wearing the sleeve on the right arm while performing the 18 everyday activities shown in Fig. 1 [23]. Each subject completed 10 rounds of all 18 activities in random order, removing and re-wearing the sleeve between rounds to emulate realistic donning variability, yielding 140 subject-round recordings. Each frame is a 20×10 matrix of 12-bit ADC readings at 50 Hz, and each round’s labels file marks the start and end frame index of every activity segment.

## B. Screening-Category Mapping

We define a deterministic, documented mapping from the 18 activity indices to three classes: neutral or acceptable (standing, back against the wall, writing with a straight back, sitting with hands on the armrests), functional or transitional (folding arms, thinking, side against the wall, arm on the baffle, hugging a doll, carrying a box), and potentially undesirable (leaning forward or back at a desk, sleeping on the table, holding one’s cheeks, using a mobile phone, writing with a hunched back, sitting leaning to one side, sitting with the arms on the legs). This mapping is a modeling construct based on the semantic content of the original labels, not a validated clinical taxonomy; no activity is labeled as an injury. Two alternative mappings are stress-tested.

## C. Feature Engineering

Let $p _ { i , j }$ denote the pressure reading at row $i \in \{ 1 , \ldots , 2 0 \}$ and column $j ~ \in ~ \{ 1 , \dots , 1 0 \}$ of a frame, $\begin{array} { r } { T \ = \ \sum _ { i , j } p _ { i , j } } \end{array}$ the total pressure, and $\begin{array} { r } { r _ { i } = \sum _ { j } p _ { i , j } , c _ { j } = \sum _ { i } p _ { i , j } } \end{array}$ the row and column marginal profiles. Global intensity features are the mean, maximum, minimum, standard deviation, median, 90th and 95th percentiles, total, and range of $\{ p _ { i , j } \}$ , plus two activation-area features recording the fraction of cells above a low and a high threshold, both estimated from training subjects only (Section II-D).

Spatial features describe the center of pressure and its spread along each axis:

$$
\bar { y } = \frac { 1 } { T } \sum _ { i } i r _ { i } , \qquad \bar { x } = \frac { 1 } { T } \sum _ { j } j c _ { j } ,\tag{1}
$$

$$
\sigma _ { y } ^ { 2 } = \frac { 1 } { T } \sum _ { i } r _ { i } ( i - \bar { y } ) ^ { 2 } , \qquad \sigma _ { x } ^ { 2 } = \frac { 1 } { T } \sum _ { j } c _ { j } ( j - \bar { x } ) ^ { 2 } .\tag{2}
$$

Asymmetry features compare pressure across the left-right (lr) and up-down (ud) halves and the four quadrants, e.g., $l r =$ $( \mathrm { R i g h t - L e f t } ) / T$ . Complexity features include the Shannon entropy of the normalized pressure distribution,

$$
H = - \sum _ { i , j } { \frac { p _ { i , j } } { T } } \log { \frac { p _ { i , j } } { T } } , \quad p _ { i , j } > 0 ,\tag{3}
$$

the concentration of the ten highest-pressure cells relative to $T ,$ and the ratio of the maximum cell to T. Two temporal features, frame-to-frame change in total pressure and frame-to-frame center-of-pressure displacement, are computed within each labeled segment. In total, 29 features are extracted per frame from 84,095 labeled frames drawn from 140 subject-round recordings, after sub-sampling every fourth frame (12.5 Hz effective rate) and capping each activity segment at 250 frames to control class imbalance and runtime.

## D. Subject-Wise Data Split and Leakage Control

Since consecutive frames from the same subject and round are highly correlated, we split at the subject level rather than the frame level: 10 subjects for training (2, 3, 4, 5, 6, 7, 8, 9, 11, 14), 2 for validation (1, 13), and 2 for held-out testing (10, 12), giving 57,596 / 13,993 / 12,506 frames per split. The split is fixed before any threshold estimation or feature computation, and the two activation-percentile thresholds are estimated from training subjects only, so no test-subject frame influences a global preprocessing statistic.

## E. Classification Models and Training Protocol

We compare a multinomial logistic regression baseline with standardized features, a random forest [17] with 300 trees, and XGBoost [18] as the primary model, all trained with class-balanced sample weights. XGBoost hyperparameters are selected via a 20-configuration randomized search with 5-fold grouped cross-validation on the training subjects (macro F1 objective, grouped by subject ID), refit with early stopping monitored on the validation subjects, and evaluated exactly once on the test subjects. XGBoost minimizes the regularized additive objective

$$
\mathcal { L } ^ { ( t ) } = \sum _ { k } l \Big ( y _ { k } , \hat { y } _ { k } ^ { ( t - 1 ) } + f _ { t } ( x _ { k } ) \Big ) + \Omega ( f _ { t } ) ,\tag{4}
$$

$$
\begin{array} { r } { \Omega ( f ) = \gamma T _ { \mathrm { l e a v e s } } + \frac { 1 } { 2 } \lambda \lVert w \rVert ^ { 2 } , } \end{array}\tag{5}
$$

where $f _ { t }$ is the tree added at boosting round t and Ω penalizes tree complexity [18]. The selected configuration used 500 boosted trees at maximum depth 4 and learning rate 0.03, with early stopping selecting the model at iteration 311.

To test whether hand-engineered features are outperformed by a learned spatial representation, we additionally train a small 2D-CNN directly on the raw 20×10 frames (two 3×3 conv blocks with batch normalization, global average pooling, two fully connected layers), using the same subject-wise split, class-frequency-weighted cross-entropy loss, Adam at $1 0 ^ { - 3 }$ and early stopping on validation macro F1.

![](images/cec3826d39e60d067f867b0ba61e8e4bfbda4bccf66de9e6a2a54de3daa4b0d2.jpg)  
Fig. 2. Representative raw $2 0 \times 1 0$ pressure frames for eight activities spanning the three screening categories.

## III. RESULTS AND DISCUSSION

## A. Qualitative and Univariate Feature Patterns

Fig. 2 shows one representative frame per activity across the three screening categories. Neutral postures (standing, writing with a straight back) spread pressure over a diffuse mid-length band with no dominant taxel, whereas undesirable postures (sleeping on the table, sitting leaning to the right, writing with a hunched back, using a mobile phone) concentrate pressure into a tight, high-intensity cluster whose location varies by activity. Carrying a box, the functional example, instead produces several smaller peaks near the sleeve edges, consistent with a two-point grip rather than sustained contact. This qualitative pattern is echoed at the univariate level: total pressure separates the neutral class from the other two, with a markedly lower median and interquartile range, while the left-right balance feature $l r \_ d i f f$ takes opposite signs for the functional and undesirable classes, indicating that these categories occupy different halves of the sleeve on average. In contrast, the vertical center of pressure cop y and the fraction of activated taxels active pct overlap heavily across all three classes.

## B. Feature Table and Class Balance

The engineered feature table contains 84,095 labeled frames and 29 features, with per-subject counts ranging from 4,489 (subject 2) to 8,911 (subject 13). Class balance is uneven by design: potentially undesirable frames make up about 48%, functional or transitional about 30%, and neutral about 22%, reflecting longer sustained segments for undesirable postures in the source recordings rather than any resampling choice. We therefore report balanced accuracy and macro F1 alongside accuracy throughout.

## C. Test-Set Performance and Confidence Intervals

Table I summarizes the three tabular models on unseen test subjects. XGBoost is the strongest classifier on every metric (XGBoost > random forest > logistic regression). Its accuracy-balanced accuracy gap narrows to 3.0 points, versus 6.2 points for logistic regression, showing it is relatively stronger on the smaller classes, and its ROC AUC of 0.942 exceeds its accuracy of 0.818, indicating well-separated class probabilities that leave room for threshold tuning. A bootstrap of 2,000 test-frame resamples gives narrow 95% intervals, accuracy [0.812, 0.825], balanced accuracy [0.780, 0.796], macro F1 [0.794, 0.808], each about ±0.01, reflecting frame-level resampling variance rather than subject-to-subject variance, which is addressed by the LOSO study in Section III-E.

TABLE I  
TEST-SET PERFORMANCE ON UNSEEN SUBJECTS (SUBJECTS 10, 12; 12,506 FRAMES)
<table><tr><td>Model</td><td>Acc.</td><td>Bal. Acc.</td><td>Macro F1</td><td>ROC AUC</td></tr><tr><td>Logistic Regression</td><td>0.730</td><td>0.668</td><td>0.681</td><td>0.873</td></tr><tr><td>Random Forest</td><td>0.793</td><td>0.737</td><td>0.762</td><td>0.933</td></tr><tr><td>XGBoost</td><td>0.818</td><td>0.788</td><td>0.801</td><td>0.942</td></tr><tr><td>TinyCNN (raw frames)</td><td>0.814</td><td>0.779</td><td>0.795</td><td></td></tr></table>

![](images/1406d19934c538a63c4977a62212843f184c83c136fd466a38d820aec58e5050.jpg)  
Fig. 3. Row-normalized confusion matrix for XGBoost on the held-out test subjects.

## D. Per-Class Confusion and Sub-Cluster Behavior

The confusion matrix ( Fig. 3) shows that the undesirable class acts as an attractor for ambiguous frames: its predicted column holds the highest diagonal value (0.92) and the two largest off-diagonal entries in the matrix (0.24 from neutral, 0.19 from functional), while direct neutral-functional confusion stays at or below 0.09. Table II shows this pattern varies by subject: subject 10 tracks the aggregate closely, while subject 12’s balanced accuracy and macro F1 sit about six points lower, an early hint of the subject variance quantified next by LOSO.

Because the undesirable class pools eight source activities, we further ask how uniformly the model recognizes them (Table III). Three activities reach recall 1.00 (sleeping on the table, holding cheeks, sitting leaning to the right) and two more exceed 0.94 (writing with a hunched back, using a mobile phone), while “sit with arms on the legs” (0.83) and “lean back at work” (0.74) are hardest, with the latter’s misclassified frames routed mostly to neutral (17.7%). The easiest activities are exactly those with a dense, concentrated contact signature, while the hardest present a more diffuse pattern; the pooled undesirable label therefore hides real heterogeneity that a single recall figure would not reveal.

TABLE II  
PER-TEST-SUBJECT XGBOOST METRICS ON THE TWO HELD-OUT SUBJECTS. SUBJECT 10’S NUMBERS ARE CLOSE TO THE AGGREGATE TEST FIGURES, WHILE SUBJECT 12’S BALANCED ACCURACY AND MACRO F1 SIT BELOW THEM, FORESHADOWING THE SUBJECT-LEVEL SPREAD QUANTIFIED BY LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION.
<table><tr><td>Subj.</td><td>Frames</td><td>Acc.</td><td>Bal. Acc.</td><td>Macro F1</td></tr><tr><td>10</td><td>6,752</td><td>0.832</td><td>0.822</td><td>0.828</td></tr><tr><td>12</td><td>5,754</td><td>0.802</td><td>0.744</td><td>0.763</td></tr></table>

TABLE III

PER-ACTIVITY BEHAVIOR OF XGBOOST INSIDE THE POOLED POTENTIALLY UNDESIRABLE CLASS ON THE TEST SUBJECTS.
<table><tr><td>Activity</td><td>Frames</td><td>Recall</td><td>%N</td><td>%U</td><td>% F</td></tr><tr><td>Lean back at work</td><td>661</td><td>0.735</td><td>17.7</td><td>73.5</td><td>8.8</td></tr><tr><td>Sit with arms on the legs</td><td>844</td><td>0.829</td><td>0.0</td><td>82.9</td><td>17.1</td></tr><tr><td>Lean forward at work</td><td>710</td><td>0.856</td><td>14.4</td><td>85.6</td><td>0.0</td></tr><tr><td>Play mobile phone</td><td>749</td><td>0.944</td><td>0.5</td><td>94.4</td><td>5.1</td></tr><tr><td>Write with a hunchback</td><td>834</td><td>0.946</td><td>5.4</td><td>94.6</td><td>0.0</td></tr><tr><td>Sleep on the table</td><td>723</td><td>1.000</td><td>0.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Hold cheeks</td><td>814</td><td>1.000</td><td>0.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Sit leaning to the right</td><td>833</td><td>1.000</td><td>0.0</td><td>100.0</td><td>0.0</td></tr></table>

## E. Leave-One-Subject-Out Cross-Validation

Retraining XGBoost with the selected hyperparameters and holding each of the 14 subjects out in turn (Fig. 4) gives a mean macro F1 of 0.770 (SD 0.059, range 0.666 to 0.887) and similarly distributed accuracy (mean 0.785, SD 0.061, range 0.660 to 0.894). The two subjects held out for the main test split (10 and 12) land at 0.834 and 0.769 macro F1, both above the mean, so the 0.801 test macro F1 reported earlier sits on the upper side of the subject distribution. The ±0.06 subject-level spread is roughly six times wider than the framelevel bootstrap interval, confirming that subject variance, not test-set noise, is the dominant source of residual uncertainty and the figure to consult when considering deployment to new wearers.

## F. Feature-Group Ablation

Retraining XGBoost on each of the six engineered feature families in isolation (Table IV) reaches 0.80 macro F1 for the full set, led by spatial features alone at 0.64, followed by global intensity (0.57), asymmetry (0.54), complexity (0.50), temporal (0.45), and activation area (0.40, the weakest). The spatial family alone recovers most of the gap from the majorityclass baseline, showing that where contact occurs on the sleeve matters more than how much of the sleeve is touched. Still, the 16-point gap between spatial alone (0.64) and the full model (0.80) shows that no single family is sufficient: XGBoost benefits from combining spatial cues with intensity, asymmetry, complexity, and temporal change, consistent with the SHAP ranking below, which shows the top features as important but not solely adequate.

Figure 9 -- Leave-one-subject-out macro F1 by held-out subject  
![](images/9f3074d8e1a22469dcbe8e15bf4535bf2cb775002bb3edf6c46f877109e3c482.jpg)  
Fig. 4. Leave-one-subject-out macro F1 for XGBoost, one bar per held-out subject; red dashed line marks the mean.

TABLE IV  
FEATURE-GROUP ABLATION ON THE HELD-OUT TEST SUBJECTS.
<table><tr><td>Feature group</td><td># feats</td><td>Acc.</td><td>Bal. Acc.</td><td>Macro F1</td></tr><tr><td>All</td><td>29</td><td>0.818</td><td>0.788</td><td>0.801</td></tr><tr><td>Spatial</td><td>6</td><td>0.688</td><td>0.639</td><td>0.643</td></tr><tr><td>Global intensity</td><td>9</td><td>0.629</td><td>0.569</td><td>0.574</td></tr><tr><td>Asymmetry</td><td>6</td><td>0.582</td><td>0.539</td><td>0.539</td></tr><tr><td>Complexity</td><td>3</td><td>0.505</td><td>0.511</td><td>0.495</td></tr><tr><td>Temporal</td><td>2</td><td>0.486</td><td>0.447</td><td>0.449</td></tr><tr><td>Activation area</td><td>3</td><td>0.460</td><td>0.397</td><td>0.405</td></tr></table>

## G. Sensitivity to the Class-Mapping Scheme

Because the three-class mapping is a design choice rather than a property of the dataset, we test two alternatives without changing anything else. Mapping A swaps “arm on the baffle” into undesirable and “sit with hands on the armrests” into functional. Mapping B moves “sitting leaning to the right” and “sit with arms on the legs” out of undesirable and into functional. Both give macro F1 close to the original (0.810 and 0.815 versus 0.801; Table V), with Mapping B also improving balanced accuracy (0.815 versus 0.788) as expected when a heterogeneous class is trimmed. The reported performance is therefore not an artifact of one arbitrary mapping choice.

## H. Explainability via SHAP

The two most influential features, cop x and var x, describe the position and spread of pressure along the 10-taxel circumferential axis and outweigh std pressure and cop y, which describe intensity variability and lengthwise position. For this sleeve geometry, where pressure wraps around the forearm therefore matters more to the model than where along its length contact occurs, plausibly because everyday postures differ more in which side of the forearm makes contact than in how far up or down it extends. The beeswarm plot (Fig. 5) shows clean color separation for var x and std pressure (high circumferential spread pushes away from undesirable; high pressure variability pushes toward it), while lower-ranked features such as lr diff and ud diff overlap on both sides of zero, indicating smaller, more context-dependent contributions. The features driving individual frame decisions match this global ranking, so the pattern is not an artifact of averaging over dissimilar frames, though exact magnitudes and occasional signs still vary frame to frame, reflecting frame-specific reweighting of the same handful of physical properties.

TABLE V  
TEST-SET XGBOOST PERFORMANCE UNDER THREE THREE-CLASS MAPPINGS EVALUATED ON THE SAME HELD-OUT SUBJECTS.
<table><tr><td>Mapping</td><td>Acc.</td><td>Bal. Acc.</td><td>Macro F1</td></tr><tr><td>Original</td><td>0.818</td><td>0.788</td><td>0.801</td></tr><tr><td>Alt. A (borderline swap)</td><td>0.841</td><td>0.790</td><td>0.810</td></tr><tr><td>Alt. B (stricter undesirable)</td><td>0.827</td><td>0.815</td><td>0.815</td></tr></table>

![](images/d0dafbdf29a0b216ea601c0f2c06e5904d4faa05255cc5e16ed3055cf9e9898e.jpg)  
Fig. 5. SHAP beeswarm plot for the potentially undesirable class. Each point is one test-set frame; horizontal position is the feature’s SHAP value and color encodes the feature’s own numerical value from low (blue) to high (red).

## I. Robustness to Simulated Donning Rotation

Evaluating the frozen XGBoost model on test frames whose 10-column axis is circularly rotated by ±1, ±2, and ±3 taxels (Table VI) shows an asymmetric, non-negligible degradation: macro F1 falls from 0.80 at shift 0 to 0.79, 0.73, and 0.62 for positive shifts of one, two, and three taxels, and to 0.72, 0.52, and 0.47 for the corresponding negative shifts. The negative direction consistently hurts more, consistent with cop x being the top SHAP feature: rotating the pressure image around the circumference directly perturbs the axis the model relies on most. A two-taxel donning offset alone would cost about seven macro-F1 points in the positive direction and nearly 28 in the negative direction, a real deployment concern that would require a wearing-orientation marker, an alignment step at donning, or rotation-augmented training.

TABLE VI  
ROBUSTNESS OF THE FROZEN XGBOOST MODEL TO SIMULATED CIRCUMFERENTIAL DONNING ROTATION.
<table><tr><td>Shift (taxels)</td><td>Acc.</td><td>Bal. Acc.</td><td>Macro F1</td></tr><tr><td>-3</td><td>0.533</td><td>0.489</td><td>0.471</td></tr><tr><td>-2</td><td>0.614</td><td>0.541</td><td>0.522</td></tr><tr><td>-1</td><td>0.761</td><td>0.706</td><td>0.721</td></tr><tr><td>0</td><td>0.818</td><td>0.788</td><td>0.801</td></tr><tr><td>+1</td><td>0.807</td><td>0.775</td><td>0.787</td></tr><tr><td>+2</td><td>0.749</td><td>0.719</td><td>0.728</td></tr><tr><td>+3</td><td>0.634</td><td>0.606</td><td>0.618</td></tr></table>

## IV. CONCLUSION

This paper presented an explainable re-analysis of the Smart-Sleeve pressure-mapping dataset for coarse postural screening. Using 29 interpretable features and a strict subjectindependent protocol, XGBoost reached 0.818 accuracy, 0.788 balanced accuracy, and 0.801 macro F1 on unseen subjects, with a bootstrap 95% interval of about ±0.01 on those metrics and a LOSO subject-level spread of about ±0.06. A simple raw-frame 2D-CNN reached essentially the same performance, so at this dataset scale the interpretable tabular pipeline is not left behind by a learned spatial representation. SHAP-based explanation, together with a feature-group ablation and a subcluster analysis, showed that circumferential spatial features contribute more than raw pressure magnitude and that the pooled undesirable class hides real heterogeneity across its source activities. A donning-rotation stress test quantified how much orientation matters for this feature set. These findings support the feasibility of coupling textile pressure sensing with explainable machine learning for low-cost, interpretable posture screening, while making the orientation dependence and the mapping-driven confusion structure explicit rather than hidden.

## V. LIMITATIONS AND FUTURE WORK

The present study has several limitations that should guide future work. The screening categories are derived heuristically from activity semantics rather than from measured biomechanical or clinical outcomes, and the dataset provides no injury or discomfort labels, so no claim of injury prediction or risk diagnosis is made or implied. The sample (14 right-handed subjects, sleeve on the right arm) also limits demographic and anatomical diversity. Future work should therefore evaluate the pipeline on a larger, more diverse subject pool; extend the rotation study to the lengthwise axis and combine rotation with training-time augmentation for orientation-tolerant variants; explore sequence models that exploit longer temporal context across activity transitions rather than withinsegment features alone; benchmark directly against recent pressure-sensing screening pipelines under identical subjectindependent protocols; and, where ethically and clinically appropriate, validate the screening categories against expert ergonomic assessment rather than activity labels alone.

## REFERENCES

[1] L. M. Castano and A. B. Flatau, “Smart fabric sensors and e-textile technologies: a review,” Smart Materials and Structures, vol. 23, no. 5, p. 053001, 2014.

[2] K. Cherenack and L. van Pieterson, “Smart textiles: Challenges and opportunities,” Journal of Applied Physics, vol. 112, no. 9, p. 091301, 2012.

[3] M. Stoppa and A. Chiolerio, “Wearable electronics and smart textiles: A critical review,” Sensors, vol. 14, no. 7, pp. 11 957–11 992, 2014.

[4] C. Mattmann, F. Clemens, and G. Troster, “Sensor for measuring strain¨ in textile,” Sensors, vol. 8, no. 6, pp. 3719–3732, 2008.

[5] J. Meyer, B. Arnrich, J. Schumm, and G. Troster, “Design and modeling¨ of a textile pressure sensor for sitting posture classification,” IEEE Sensors Journal, vol. 10, no. 8, pp. 1391–1398, 2010.

[6] W. Xu, M.-C. Huang, N. Amini, L. He, and M. Sarrafzadeh, “eCushion: A textile pressure sensor array design and calibration for sitting posture analysis,” IEEE Sensors Journal, vol. 13, no. 10, pp. 3926–3934, 2013.

[7] J. Cheng, B. Zhou, K. Kunze, C. C. Rheinlander, S. Wille, N. Wehn,¨ J. Weppner, and P. Lukowicz, “Activity recognition and nutrition monitoring in every day situations with a textile capacitive neckband,” in Proc. ACM Conf. Pervasive and Ubiquitous Computing Adjunct Publication (UbiComp), 2013, pp. 155–158.

[8] J. Cheng, O. Amft, G. Bahle, and P. Lukowicz, “Designing sensitive wearable capacitive sensors for activity recognition,” IEEE Sensors Journal, vol. 13, no. 10, pp. 3935–3947, 2013.

[9] M. Sundholm, J. Cheng, B. Zhou, A. Sethi, and P. Lukowicz, “Smartmat: Recognizing and counting gym exercises with low-cost resistive pressure sensing matrix,” in Proc. ACM Int. Joint Conf. Pervasive and Ubiquitous Computing (UbiComp), 2014, pp. 373–382.

[10] O. D. Lara and M. A. Labrador, “A survey on human activity recognition using wearable sensors,” IEEE Communications Surveys & Tutorials, vol. 15, no. 3, pp. 1192–1209, 2013.

[11] A. Bulling, U. Blanke, and B. Schiele, “A tutorial on human activity recognition using body-worn inertial sensors,” ACM Computing Surveys, vol. 46, no. 3, p. 33, 2014.

[12] J. Wang, Y. Chen, S. Hao, X. Peng, and L. Hu, “Deep learning for sensorbased activity recognition: A survey,” Pattern Recognition Letters, vol. 119, pp. 3–11, 2019.

[13] K. Chen, D. Zhang, L. Yao, B. Guo, Z. Yu, and Y. Liu, “Deep learning for sensor-based human activity recognition: Overview, challenges, and opportunities,” ACM Computing Surveys, vol. 54, no. 4, p. 77, 2021.

[14] M. Straczkiewicz, P. James, and J.-P. Onnela, “A systematic review of smartphone-based human activity recognition methods for health research,” npj Digital Medicine, vol. 4, p. 148, 2021.

[15] R. Zemp, M. Tanadini, S. Pluss, K. Schn ¨ uriger, N. B. Singh, W. R.¨ Taylor, and S. Lorenzetti, “Application of machine learning approaches for classifying sitting posture based on force and acceleration sensors,” BioMed Research International, vol. 2016, p. 5978489, 2016.

[16] M. B. Pouyan, J. Birjandtalab, M. Heydarzadeh, M. Nourani, and S. Ostadabbas, “A pressure map dataset for posture and subject analytics,” in Proc. IEEE EMBS Int. Conf. Biomedical and Health Informatics (BHI), 2017, pp. 65–68.

[17] L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

[18] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD), 2016, pp. 785–794.

[19] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 4765–4774.

[20] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, “From local explanations to global understanding with explainable AI for trees,” Nature Machine Intelligence, vol. 2, no. 1, pp. 56–67, 2020.

[21] A. Adadi and M. Berrada, “Peeking inside the black-box: A survey on explainable artificial intelligence (XAI),” IEEE Access, vol. 6, pp. 52 138–52 160, 2018.

[22] F. K. Dosiloviˇ c, M. Br´ ciˇ c, and N. Hlupi´ c, “Explainable artificial in-´ telligence: A survey,” in Proc. 41st Int. Convention on Information and Communication Technology, Electronics and Microelectronics (MIPRO), 2018, pp. 0210–0215.

[23] G. Xu, W. Deng, and J. Cheng, “Smart-Sleeve dataset based on pressure mapping smart textile,” GitHub repository, University of Science and Technology of China, 2023, available: https://github.com/xghgithub/ Smart-Sleeve-Dataset.