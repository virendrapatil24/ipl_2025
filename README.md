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
│   │   ├── raw_data/
│   │   ├── processed_data/
│   │   └── cleaned_data/
│   └── vector_store/
│   └── scripts/
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

## Data Storage Approach

This project uses a file-based data storage approach rather than a traditional database:

- All data is stored at the repository level in the `backend/data/` directory
- Raw data is stored in `backend/data/raw_data/`
- Processed data is stored in `backend/data/processed_data/`
- Cleaned data is stored in `backend/data/cleaned_data/`
- Vector embeddings are stored in `backend/vector_store/`

This approach was chosen for:

- Ease of setup and deployment
- Simplicity in data versioning
- No need for database configuration
- Quick iteration during development

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

## Future Scope

- **Database Integration**: Replace file-based storage with a proper database (PostgreSQL, MongoDB) for better scalability and performance
- **Real-time Data Updates**: Implement a system to fetch and update data in real-time from cricket APIs
- **User Accounts**: Add user authentication and personalized predictions
- **Advanced Analytics**: Incorporate machine learning models for more accurate predictions
- **Mobile App**: Develop a mobile application for on-the-go access
- **Historical Match Replay**: Add feature to analyze past matches with detailed statistics
- **Fantasy League Integration**: Connect with popular fantasy cricket platforms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
