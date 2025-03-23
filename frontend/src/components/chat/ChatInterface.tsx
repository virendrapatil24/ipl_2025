"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ModelSelector } from "./ModelSelector"
import { MessageList } from "./MessageList"
import { MessageInput } from "./MessageInput"

export type Message = {
    role: "user" | "assistant"
    content: string
}

export function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([])
    const [isLoading, setIsLoading] = useState(false)
    const [selectedModel, setSelectedModel] = useState("gpt-4")

    const handleSendMessage = async (content: string) => {
        const newMessage: Message = { role: "user", content }
        setMessages((prev) => [...prev, newMessage])
        setIsLoading(true)

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: content,
                    model: selectedModel,
                }),
            })

            const data = await response.json()
            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: data.response },
            ])
        } catch (error) {
            console.error("Error sending message:", error)
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <Card className="w-full max-w-4xl mx-auto h-[calc(100vh-4rem)]">
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <span>IPL Fantasy Predictor</span>
                    <ModelSelector
                        value={selectedModel}
                        onChange={setSelectedModel}
                    />
                </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col h-[calc(100%-5rem)]">
                <MessageList messages={messages} isLoading={isLoading} />
                <MessageInput onSend={handleSendMessage} />
            </CardContent>
        </Card>
    )
} 