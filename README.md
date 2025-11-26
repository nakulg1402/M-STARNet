# M-STARNet
Emotion recognition is essential for brain-computer interac-
tions (BCIs) and affective computing, enabling machines to interpret and
respond to human emotions for more natural and effective interaction. Al-
though existing methods perform well, their multi-class emotion classifica-
tion accuracy remains limited. This paper presents a multimodal spatio-
temporal attentive representation network (M-STARNet) for the classifi-
cation of human emotions. The proposed architecture employs electroen-
cephalogram (EEG) and electrooculogram (EOG) signals from the pub-
licly available SEED VII dataset. Transformer encoders with positional
encodings (PE) are used to extract spatio-temporal and ocular features.
These features are then stacked and passed through the random modality
dropout module to reduce modality-specific bias and improve multimodal
cross-subject emotion classification. M-STARNet achieves substantially
higher multi-class accuracy than existing methods, attaining 54.68% in
subject-dependent and 92.55% in subject independent scenarios. This per-
formance demonstrates improved generalization across multiple emotion
states and highlights the potential of the proposed framework to support
the development of reliable affect-based systems.
The code is accessible at: github.com/nakulg1402/M-STARNet
