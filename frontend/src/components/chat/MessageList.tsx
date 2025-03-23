import { Message } from "./ChatInterface"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Separator } from "@/components/ui/separator"
import ReactMarkdown from "react-markdown"

interface MessageListProps {
    messages: Message[]
    isLoading: boolean
}

export function MessageList({ messages, isLoading }: MessageListProps) {
    return (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
                <div key={index} className="flex gap-4">
                    <Avatar>
                        <AvatarImage
                            src={
                                message.role === "user"
                                    ? "/user-avatar.png"
                                    : "/assistant-avatar.png"
                            }
                        />
                        <AvatarFallback>
                            {message.role === "user" ? "U" : "A"}
                        </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 space-y-2">
                        <div className="rounded-lg bg-muted p-4">
                            <ReactMarkdown>{message.content}</ReactMarkdown>
                        </div>
                        <Separator />
                    </div>
                </div>
            ))}
            {isLoading && (
                <div className="flex gap-4">
                    <Avatar>
                        <AvatarImage src="/assistant-avatar.png" />
                        <AvatarFallback>A</AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                        <div className="rounded-lg bg-muted p-4">
                            <div className="animate-pulse">Thinking...</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
} 