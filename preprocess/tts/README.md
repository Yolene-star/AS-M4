# Audio Instruction Synthesis

1. **Audio Prompt Preparation**

   Download the audio from the [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K).

2. **Prepare the TTS Tool**

   Option 1: [ChatTTS](https://github.com/2noise/ChatTTS)

   Option 2: [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (recommend)

   CosyVoice is not published as a PyPI package named `cosyvoice.cli`. Since this workspace keeps a local checkout in `third_party/CosyVoice`, make sure that directory is on `PYTHONPATH` before running the preprocessing script:

   ```bash
   export PYTHONPATH="$(pwd)/third_party/CosyVoice:${PYTHONPATH:-}"
   ```

   If you use a different checkout path, make sure the `cosyvoice` Python package is importable before running the preprocessing script.

   The preprocessing script expects the CosyVoice base model to be available as `iic/CosyVoice-300M` by default. You can override it with `COSYVOICE_MODEL_DIR` if needed.

   It also expects the following local assets under the project root:

   - `voiceassistant.json`
   - `m4-it-qwen.json`
   - `VoiceAssistant-400K/audios/`

3. **Randomly Select the Audio Prompt and Synthesize the Audio Instruction**

   *Note: Before running the script, check the directories in the script.*

   ```bash
   process_cosyvoice.sh
   ```