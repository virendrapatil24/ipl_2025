/**
 * Application configuration
 *
 * This file contains environment variables and configuration settings
 * for the application.
 */

// Server availability flag
export const SERVER_AVAILABLE =
  process.env.NEXT_PUBLIC_SERVER_AVAILABLE === "true";

// API endpoint
export const API_ENDPOINT =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Fallback message when server is unavailable
export const SERVER_UNAVAILABLE_MESSAGE =
  "Our servers are currently down. You can run the application locally by following these steps:\n\n" +
  "1. Clone the repository: git clone https://github.com/yourusername/ipl_2025.git\n" +
  "2. Install dependencies: cd ipl_2025 && npm install\n" +
  "3. Start the development server: npm run dev\n\n" +
  "Alternatively, you can check back later when our servers are back online.";
