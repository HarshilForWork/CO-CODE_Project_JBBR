"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "../../components/ui/input"
import { Label } from "../../components/ui/label"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { uploadPDF, getNextFlashcard, type Flashcard } from "../../lib/api"

const demoFlashcards = [
  {
    topic: "Photosynthesis",
    question: "What is photosynthesis?",
    answer:
      "Photosynthesis is the process by which plants use sunlight, water and carbon dioxide to produce oxygen and energy in the form of sugar.",
  },
  {
    topic: "Photosynthesis Components",
    question: "What are the main components needed for photosynthesis?",
    answer: "The main components needed for photosynthesis are: sunlight, water, carbon dioxide, and chlorophyll.",
  },
  {
    topic: "Photosynthesis Location",
    question: "Where does photosynthesis take place in the plant?",
    answer:
      "Photosynthesis primarily takes place in the leaves of plants, specifically in the chloroplasts of plant cells.",
  },
  {
    topic: "Chlorophyll",
    question: "What is the role of chlorophyll in photosynthesis?",
    answer:
      "Chlorophyll is the pigment that gives plants their green color and is responsible for absorbing light energy used in photosynthesis.",
  },
  {
    topic: "Photosynthesis Stages",
    question: "What are the two main stages of photosynthesis?",
    answer:
      "The two main stages of photosynthesis are the light-dependent reactions and the light-independent reactions (Calvin cycle).",
  },
]

export default function FlashcardsPage() {
  const [currentCard, setCurrentCard] = useState(0)
  const [isFlipped, setIsFlipped] = useState(false)
  const [isDemoMode, setIsDemoMode] = useState(true)
  const [file, setFile] = useState<File | null>(null)
  const [numCards, setNumCards] = useState<number>(5)
  const [specificTopic, setSpecificTopic] = useState<string>("")
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [flashcard, setFlashcard] = useState<Flashcard | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0])
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a PDF file.")
      return
    }

    if (numCards < 1 || numCards > 20) {
      setError("Please enter a number between 1 and 20.")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const data = await uploadPDF(
        file, 
        numCards, 
        specificTopic || undefined
      )
      setSessionId(data.session_id)
      setFlashcard(data.flashcard)
      setIsDemoMode(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload PDF.")
    } finally {
      setLoading(false)
    }
  }

  const handleNextFlashcard = async () => {
    setIsFlipped(false)
    if (isDemoMode) {
      setCurrentCard((prev) => (prev + 1) % demoFlashcards.length)
    } else {
      if (!sessionId) {
        setError("No session ID found. Please upload a PDF first.")
        return
      }

      setLoading(true)
      setError(null)

      try {
        const data = await getNextFlashcard(sessionId)
        if (data.flashcard) {
          setFlashcard(data.flashcard)
        } else {
          setError("No more flashcards available.")
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch the next flashcard.")
      } finally {
        setLoading(false)
      }
    }
  }

  const handlePreviousFlashcard = () => {
    if (isDemoMode) {
      setIsFlipped(false)
      setCurrentCard((prev) => (prev - 1 + demoFlashcards.length) % demoFlashcards.length)
    }
  }

  const toggleFlip = () => {
    setIsFlipped(!isFlipped)
  }

  const switchToDemo = () => {
    setIsDemoMode(true)
    setCurrentCard(0)
    setIsFlipped(false)
    setSessionId(null)
    setFlashcard(null)
    setError(null)
    setFile(null)
    setNumCards(5)
    setSpecificTopic("")
  }

  const switchToPDFUpload = () => {
    setIsDemoMode(false)
    setCurrentCard(0)
    setIsFlipped(false)
    setSessionId(null)
    setFlashcard(null)
    setError(null)
  }

  const getCurrentFlashcard = () => {
    if (isDemoMode) {
      return demoFlashcards[currentCard]
    }
    return flashcard
  }

  const currentFlashcard = getCurrentFlashcard()

  return (
    <div className="min-h-screen overflow-hidden bg-gray-50">
      <div className="container max-w-4xl mx-auto px-4 py-10">
        <h1 className="text-3xl font-bold text-center mb-10">
          QuizEasy - Flashcard Generator
        </h1>

        <div className="flex justify-center mb-10 space-x-4">
          <Button 
            onClick={switchToDemo} 
            variant={isDemoMode ? "default" : "outline"} 
            className="w-40"
          >
            Demo Flashcards
          </Button>
          <Button 
            onClick={switchToPDFUpload} 
            variant={!isDemoMode ? "default" : "outline"} 
            className="w-40"
          >
            Upload PDF
          </Button>
        </div>

        {!isDemoMode && (
          <div className="mb-10">
            <div className="grid gap-6 max-w-xl mx-auto bg-white p-6 rounded-lg shadow-sm">
              <div className="space-y-2">
                <Label htmlFor="pdf-upload">
                  Upload PDF
                </Label>
                <Input
                  id="pdf-upload"
                  type="file"
                  accept="application/pdf"
                  onChange={handleFileChange}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="num-cards">
                  Number of Flashcards (1-20)
                </Label>
                <Input
                  id="num-cards"
                  type="number"
                  min="1"
                  max="20"
                  value={numCards}
                  onChange={(e) => setNumCards(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="specific-topic">
                  Specific Topic (Optional)
                </Label>
                <Input
                  id="specific-topic"
                  type="text"
                  value={specificTopic}
                  onChange={(e) => setSpecificTopic(e.target.value)}
                  placeholder="Leave empty for automatic topic detection"
                  className="w-full"
                />
              </div>

              <Button 
                onClick={handleUpload} 
                disabled={loading || !file}
                className="w-full"
              >
                {loading ? "Generating Flashcards..." : "Generate Flashcards"}
              </Button>
            </div>
          </div>
        )}

        {error && (
          <div className="mb-6 text-center">
            <p className="text-red-500 bg-red-50 p-3 rounded-lg">{error}</p>
          </div>
        )}

        {currentFlashcard && (
          <div className="relative mb-6">
            <Card className="w-full aspect-[3/1.5] perspective">
              <CardContent
                className={`absolute w-full h-full transition-transform duration-500 preserve-3d cursor-pointer ${
                  isFlipped ? "rotate-y-180" : ""
                }`}
                onClick={toggleFlip}
              >
                {/* Front of card */}
                <div className="absolute inset-0 backface-hidden">
                  <div className="flex flex-col items-center justify-center h-full p-6 text-center">
                    <p className="text-sm text-muted-foreground mb-4">
                      {currentFlashcard.topic}
                    </p>
                    <p className="text-xl font-medium">
                      {currentFlashcard.question}
                    </p>
                  </div>
                </div>
                {/* Back of card */}
                <div className="absolute inset-0 backface-hidden rotate-y-180">
                  <div className="flex items-center justify-center h-full p-6 text-center bg-primary text-primary-foreground rounded-lg">
                    <p className="text-lg">{currentFlashcard.answer}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="mt-6 flex items-center justify-between">
              <div className="flex gap-2">
                <Button 
                  onClick={handlePreviousFlashcard} 
                  disabled={loading || !isDemoMode} 
                  className="w-40"
                >
                  <ChevronLeft className="mr-2 h-4 w-4" />
                  Previous
                </Button>
                <Button 
                  onClick={handleNextFlashcard} 
                  disabled={loading} 
                  className="w-40"
                >
                  Next
                  <ChevronRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
              <p className="text-sm text-muted-foreground">
                {isDemoMode
                  ? `Card ${currentCard + 1} of ${demoFlashcards.length}`
                  : flashcard
                    ? "Generated Flashcard"
                    : "No flashcard available"}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}