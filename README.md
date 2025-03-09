# Making an end-to-end Speech Recognition and Voice AI Assistant (Khabib Nurmagomedov former UFC Champ) to help me with my job search!

Current signal flow -> PortAudio mic capture (C++) -> Noise reduction using Weiner Filtering -> Voice Activity Detection (Python, Silero) -> Wake Word detection (custom trained model on wake word "Brother Khabib")
-> Speech Recognition module (Probably Coqui Speech To Text) -> LLM (OpenAI) -> Speech Synthesis (Coqui Text To Speech)

Beamforming is a possibility, but trying to focus on getting the system up and running first.
