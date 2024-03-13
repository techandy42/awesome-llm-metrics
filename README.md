# awesome-llm-metrics
An open-source framework that makes evaluating LLMs &amp; prompt engineering x10 easier!

### Conda Environment
- To run the program locally, please copy the conda environment and activate it.
```bash
!conda env create -f environment.yml
!conda activate env
```

### Running Program
- To run individual models, please run `python3 -m models.openai_module` (or other module names).

### API Keys
- Please create a `.env` file by copying and changing the name of `.env.example` file, then following the instructions in the `.env` files to replace the placeholders with your API keys.

### Gemini API
- Google AI Studio is not supported in Canada, thus you either need a VPN to change your IP address to the US, or use the Vertex AI SDK instead.
- If you have a VPN, please turn it to US for both your browser and device. Afterwards, you can get the API key on [Google AI Studio](https://aistudio.google.com/app/apikey), write the key to `GOOGLE_API_KEY` in `.env`, then use the `google_generative_ai_module.py` file.
- Otherwise, you need to download the gcloud cli to login to your Google account via CLI tool. Please follow the below instructions to install and login with gcloud cli.
- After logging in with gcloud cli, you can write the Google Project ID to `GOOGLE_PROJECT_ID`, then start using the `google_vertex_ai_module.py` file.
```bash
# Note: This is for Linux.
# Navigate to your home directory
!cd ~
!curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-468.0.0-linux-x86_64.tar.gz
!tar -xf google-cloud-cli-468.0.0-linux-x86_64.tar.gz
!./google-cloud-sdk/install.sh
!gcloud auth login
# Please refer to Official Documentation for full details: https://cloud.google.com/sdk/docs/install-sdk
```
