"use client";

import { useState } from "react";
import { uploadPDF, getNextFlashcard, Flashcard } from "C:/PF/Projects/CO-CODE/quizeasy-frontend/lib/api";

export default function FlashcardsPage() {
    const [file, setFile] = useState<File | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [flashcard, setFlashcard] = useState<Flashcard | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            setFile(event.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            alert("Please select a PDF file.");
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const data = await uploadPDF(file, 5); // Request 5 flashcards
            setSessionId(data.session_id);
            setFlashcard(data.flashcard);
        } catch (err) {
            setError("Failed to upload PDF.");
        } finally {
            setLoading(false);
        }
    };

    const handleNextFlashcard = async () => {
        if (!sessionId) {
            alert("No session ID found. Please upload a PDF first.");
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const data = await getNextFlashcard(sessionId);
            setFlashcard(data.flashcard);
        } catch (err) {
            setError("No more flashcards available.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-lg mx-auto p-6">
            <h1 className="text-2xl font-bold mb-4">QuizEasy - Flashcard Generator</h1>

            <input type="file" accept="application/pdf" onChange={handleFileChange} />
            <button
                onClick={handleUpload}
                className="bg-blue-500 text-white px-4 py-2 mt-2 rounded"
                disabled={loading}
            >
                {loading ? "Uploading..." : "Upload PDF"}
            </button>

            {error && <p className="text-red-500 mt-2">{error}</p>}

            {flashcard && (
                <div className="mt-6 border p-4 rounded-lg shadow">
                    <h2 className="text-lg font-semibold">Topic: {flashcard.topic}</h2>
                    <p><strong>Q:</strong> {flashcard.question}</p>
                    <p><strong>A:</strong> {flashcard.answer}</p>
                    <button
                        onClick={handleNextFlashcard}
                        className="bg-green-500 text-white px-4 py-2 mt-4 rounded"
                        disabled={loading}
                    >
                        {loading ? "Loading..." : "Next Flashcard"}
                    </button>
                </div>
            )}
        </div>
    );
}
