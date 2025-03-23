import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send } from "lucide-react"

interface MessageInputProps {
    onSend: (message: string) => void
}

export function MessageInput({ onSend }: MessageInputProps) {
    const [message, setMessage] = useState("")

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        if (message.trim()) {
            onSend(message)
            setMessage("")
        }
    }

    return (
        <form onSubmit={handleSubmit} className="flex gap-2 p-4 border-t">
            <Input
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Enter match details (e.g., Team 1 vs Team 2 at xyz venue with both sides playing 11 and pitch report)"
                className="flex-1"
            />
            <Button type="submit" size="icon">
                <Send className="h-4 w-4" />
            </Button>
        </form>
    )
} 