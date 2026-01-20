# Open WebUI Integration

This document explains how to connect your **Marketing Agent** to [Open WebUI](https://openwebui.com/).

## 1. Start the Backend Server

Ensure your `.env` is configured with your `OPENAI_API_KEY`. 

### Attachment Support (Optional but Recommended)
To allow the agent to automatically analyze files attached in the chat, you need to provide the backend with access to Open WebUI's API:

1. In Open WebUI, go to **Settings > Account > API Keys**.
2. Create a new API Key and copy it.
3. Update your `.env` file:
   ```env
   OPEN_WEBUI_URL=http://your-open-webui-url:3000
   OPEN_WEBUI_API_KEY=your_key_here
   ```


Then run the server:
```bash
python -m src.ui.server
```

## 2. Automatic Document Analysis

The system automatically determines whether a file is a **Briefing** or an **Edital** based on its filename.

### How it works in Chat
When you attach a file in an Open WebUI chat session:
1. Open WebUI sends the file metadata to the backend.
2. The backend fetches the file bytes using the API Key provided in `.env`.
3. It classifies the file by name (Briefing vs. Edital).
4. It performs the analysis and returns the Markdown summary as the chat response.
5. The summary is also saved to `data/summaries/`.

### Keywords for Classification
- **Briefing**: `briefing`, `brief`, `brf`
- **Edital (Bidding)**: `edital`, `tender`, `rfp`, `bid`, `procurement`

## 3. Configure Open WebUI Connections

1. Open WebUI > **Settings** > **Connections**.
2. Under **OpenAI API**, click the `+` button.
3. **API Base URL**: `http://localhost:8000/v1` (Ensure it ends in `/v1`).
4. **API Key**: `anything` (Required by UI, but ignored by backend).
5. Click **Save**.
6. Select the `marketing-rag-agent` model from the chat interface.

## 4. Manual Analysis via CLI
You can also run the analysis locally without the API:
```bash
python -m src.agents.auto_analyzer "path/to/my_briefing.pdf"
```

## Troubleshooting
- **"Não foi possível acessar o arquivo"**: Check if `OPEN_WEBUI_URL` and `OPEN_WEBUI_API_KEY` are correct in `.env`.
- **404 Errors**: Ensure the API Base URL in Open WebUI includes the `/v1` suffix.
