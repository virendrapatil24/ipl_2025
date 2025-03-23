# IPL Fantasy Predictor RAG

A Retrieval-Augmented Generation (RAG) system for IPL match predictions and analysis.

## Project Structure

```
ipl_2025/
├── backend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   └── schemas/
│   │   ├── data_processing/
│   │   ├── llm/
│   │   ├── rag/
│   │   ├── config/
│   │   └── utils/
│   ├── tests/
│   ├── data/
│   └── vector_store/
└── frontend/
    ├── src/
    │   ├── app/
    │   ├── components/
    │   └── lib/
    ├── public/
    └── tests/
```

## Features

- Real-time match predictions using historical data
- Team and player performance analysis
- Venue-specific insights
- Head-to-head statistics
- Interactive chat interface
- Support for multiple LLM providers (OpenAI, Anthropic, Ollama)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ipl_2025.git
cd ipl_2025
```

2. Set up the backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

4. Set up the frontend:

```bash
cd ../frontend
npm install
```

## Running the Application

1. Start the backend server:

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn src.api.main:app --reload
```

2. Start the frontend development server:

```bash
cd frontend
npm run dev
```

3. Open your browser and navigate to `http://localhost:3000`

## Testing

Run backend tests:

```bash
cd backend
pytest
```

Run frontend tests:

```bash
cd frontend
npm test
```

## API Documentation

Once the backend server is running, you can access:

- API documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
