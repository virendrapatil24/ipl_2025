"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ModelSelector } from "./ModelSelector";
import { MessageList } from "./MessageList";
import { MessageInput } from "./MessageInput";
import {
  SERVER_AVAILABLE,
  SERVER_UNAVAILABLE_MESSAGE,
  API_ENDPOINT,
} from "@/lib/config";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

export type Message = {
  role: "user" | "assistant";
  content: string;
};

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("gpt-4");
  const [showServerAlert, setShowServerAlert] = useState(!SERVER_AVAILABLE);

  const handleSendMessage = async (content: string) => {
    const newMessage: Message = { role: "user", content };
    setMessages((prev) => [...prev, newMessage]);
    setIsLoading(true);

    try {
      if (!SERVER_AVAILABLE) {
        setShowServerAlert(true);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: SERVER_UNAVAILABLE_MESSAGE },
        ]);
        return;
      }

      const response = await fetch(`${API_ENDPOINT}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          model: selectedModel,
        }),
      });

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, there was an error processing your request. Please try again later.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto h-[calc(100vh-4rem)]">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>IPL Fantasy Predictor</span>
          <ModelSelector value={selectedModel} onChange={setSelectedModel} />
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col h-[calc(100%-5rem)]">
        {showServerAlert && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Server Unavailable</AlertTitle>
            <AlertDescription>
              Our servers are currently down. You can run the application
              locally by following the instructions in the chat.
            </AlertDescription>
          </Alert>
        )}
        <MessageList messages={messages} isLoading={isLoading} />
        <MessageInput onSend={handleSendMessage} />
      </CardContent>
    </Card>
  );
}
