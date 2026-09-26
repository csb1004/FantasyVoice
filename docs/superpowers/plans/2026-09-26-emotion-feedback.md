# Sequential emotion supervision and TTS feedback

User-confirmed design (September 26): first train the audio emotion analyzer on
real audio and the existing emotion table. Finish that stage, then freeze it.
Train the generator with existing transcript text, emotion/style labels and
character IDs. Add generated-audio emotion error to existing TTS losses.
No alternating analyzer updates during TTS training, and no RL/value network.

Implementation:
1. scripts/emotion_feedback.py: original pinned emotion2vec large adapter,
   differentiable text-prior synthesis, frozen-analyzer feedback objective.
2. scripts/emotion_training.py: supervised analyzer loop, independent validation,
   baseline selection, resumable optimizer/RNG and completed-stage handoff.
3. scripts/colab_train_emotion.py + notebook 11: original waveform cache and
   emotion labels. No generated audio used to update the analyzer.
4. scripts/colab_emotion_feedback.py + notebook 12: completed 09 generator,
   completed 11 analyzer. Predictor stays frozen; label conditions drive TTS.
5. Tests for actual Melo prior gradients, frozen analyzer/waveform gradients,
   supervised analyzer update, exact resume, pending validation and identity
   rejection. Real pinned analyzer smoke test separately from unit tests.
6. Documentation, notebook/script parity, full regression suite, code ZIP,
   authorized GitHub publication. Existing src and 08/09 remain unchanged.

Provisional visible defaults: one utterance per microbatch, accumulation 4,
one epoch per stage, analyzer lr 1e-6, TTS lr 1e-5, feedback weight 1.
Duration limit 15 seconds; log excluded counts and reject a lost character.
Guard generated duration before attention allocation. No silent truncation.

11 best selection uses full held-out KL and includes the pretrained baseline;
existing labels are pseudo labels from that model, so improvement is not assumed.
12 best selection retains full validation mel for 10 compatibility. Report
generated/real emotion KL with the fixed analyzer separately. Rounded durations
do not pass this emotion gradient; existing duration loss remains active.
Full model fine-tuning/full-waveform differentiation may exceed T4 memory.
