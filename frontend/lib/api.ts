
export interface Flashcard {
    topic: string;
    question: string;
    answer: string;
}

export interface UploadResponse {
    session_id: string;
    flashcard: Flashcard | null;
}

export interface NextFlashcardResponse {
    session_id: string;
    flashcard: Flashcard | null;
}

export const uploadPDF = async (
    file: File, 
    numCards: number, 
    specificTopic?: string
): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_cards", numCards.toString());
    if (specificTopic) {
        formData.append("specific_topic", specificTopic);
    }

    const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to upload PDF");
    }

    return response.json();
};

export const getNextFlashcard = async (sessionId: string): Promise<NextFlashcardResponse> => {
    const response = await fetch("http://localhost:8000/next-flashcard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "No more flashcards available.");
    }

    return response.json();
};