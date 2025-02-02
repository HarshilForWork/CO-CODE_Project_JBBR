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

// Upload PDF and get session ID + first flashcard
export const uploadPDF = async (file: File, numCards: number): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_cards", numCards.toString());

    const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error("Failed to upload PDF");
    }

    return response.json();
};

// Fetch the next flashcard from FastAPI
export const getNextFlashcard = async (sessionId: string): Promise<NextFlashcardResponse> => {
    const response = await fetch("http://localhost:8000/next-flashcard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
    });

    if (!response.ok) {
        throw new Error("No more flashcards available.");
    }

    return response.json();
};
