"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts"
import { PerformanceAnalysis } from "../../components/performance-analysis"
import { useTheme } from "next-themes"
import { CheckCircle, XCircle } from "lucide-react"

// Import the API functions
import { getNextQuizQuestion } from "../../services/api"

// Demo questions about photosynthesis
const demoQuestions = [
  {
    id: 1,
    question: "What is the primary function of photosynthesis?",
    options: [
      "Converting light energy to chemical energy",
      "Breaking down glucose",
      "Cellular respiration",
      "Nitrogen fixation",
    ],
    correctAnswer: 0,
  },
  {
    id: 2,
    question: "Which organelle is responsible for photosynthesis in plants?",
    options: ["Mitochondria", "Chloroplast", "Nucleus", "Golgi apparatus"],
    correctAnswer: 1,
  },
  // Add more demo questions here...
]

interface QuizQuestion {
  id: number
  question: string
  options: string[]
  correctAnswer: number
}

export default function QuizPage() {
  const [questions, setQuestions] = useState<QuizQuestion[]>([])
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState<number[]>([])
  const [showResults, setShowResults] = useState(false)
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)
  const [questionStartTime, setQuestionStartTime] = useState<number>(Date.now())
  const [questionTimes, setQuestionTimes] = useState<number[]>([])
  const [isDemo, setIsDemo] = useState(false)
  const { theme } = useTheme()
  const router = useRouter()

  useEffect(() => {
    const contentMode = localStorage.getItem("contentMode")
    if (contentMode === "demo") {
      setQuestions(demoQuestions)
      setIsDemo(true)
    } else {
      const storedContent = localStorage.getItem("generatedContent")
      if (storedContent) {
        const parsedContent = JSON.parse(storedContent)
        if (parsedContent.type === "quiz") {
          setQuestions([parsedContent.content])
          fetchNextQuestion()
        } else {
          router.push("/flashcards")
        }
      } else {
        router.push("/quiz-options")
      }
    }
    setQuestionStartTime(Date.now())
  }, [router])

  useEffect(() => {
    setQuestionStartTime(Date.now())
  }, []) //Corrected useEffect dependency array

  const fetchNextQuestion = async () => {
    if (!isDemo) {
      try {
        const nextQuestion = await getNextQuizQuestion(
          answers.length > 0 ? answers[answers.length - 1] === questions[currentQuestion].correctAnswer : undefined,
        )
        setQuestions((prevQuestions) => [...prevQuestions, nextQuestion])
      } catch (error) {
        console.error("Failed to fetch next question:", error)
      }
    }
  }

  const handleAnswer = (answerIndex: number) => {
    setSelectedAnswer(answerIndex)
  }

  const handleNext = async () => {
    if (selectedAnswer !== null) {
      const endTime = Date.now()
      const timeTaken = (endTime - questionStartTime) / 1000
      setQuestionTimes((prevTimes) => [...prevTimes, timeTaken])
      setAnswers((prevAnswers) => [...prevAnswers, selectedAnswer])
      setSelectedAnswer(null)

      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion((prevQuestion) => prevQuestion + 1)
        setQuestionStartTime(Date.now())
      } else {
        if (!isDemo) {
          await fetchNextQuestion()
        }
        setShowResults(true)
      }
    }
  }

  const handlePrevious = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion((prevQuestion) => prevQuestion - 1)
      setAnswers((prevAnswers) => prevAnswers.slice(0, -1))
      setQuestionTimes((prevTimes) => prevTimes.slice(0, -1))
      setSelectedAnswer(null)
    }
  }

  const calculateScore = () => {
    let correct = 0
    answers.forEach((answer, index) => {
      if (answer === questions[index].correctAnswer) correct++
    })
    return correct
  }

  if (questions.length === 0) {
    return <div>Loading...</div>
  }

  if (showResults) {
    const score = calculateScore()
    const correctAnswers = score
    const incorrectAnswers = questions.length - score
    const data = [
      { name: "Correct", value: correctAnswers },
      { name: "Incorrect", value: incorrectAnswers },
    ]
    const COLORS = ["hsl(var(--primary))", theme === "dark" ? "hsl(var(--background))" : "hsl(var(--foreground))"]
    const totalTime = questionTimes.reduce((sum, time) => sum + time, 0)

    const CustomTooltip = ({ active, payload }: any) => {
      if (active && payload && payload.length) {
        const data = payload[0].payload
        return (
          <div className="bg-background p-2 border rounded shadow">
            <p className="font-semibold">{`${data.name}: ${data.value}`}</p>
            <p>{`${((data.value / questions.length) * 100).toFixed(1)}%`}</p>
          </div>
        )
      }
      return null
    }

    return (
      <div className="container max-w-4xl py-12">
        <Card>
          <CardHeader>
            <CardTitle>Quiz Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={data} cx="50%" cy="50%" labelLine={false} outerRadius={80} fill="#8884d8" dataKey="value">
                    {data.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="hsl(var(--border))" />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-4">
              <h3 className="text-2xl font-bold">Your Score: {((score / questions.length) * 100).toFixed(1)}%</h3>
              <div className="space-y-2">
                {questions.map((q, index) => (
                  <div
                    key={q.id}
                    className={`p-4 rounded-lg ${
                      answers[index] === q.correctAnswer ? "bg-primary/10 border-primary" : "bg-muted/50 border-muted"
                    }`}
                  >
                    <p className="font-medium">{q.question}</p>
                    <p className="text-sm text-muted-foreground flex items-center">
                      {answers[index] === q.correctAnswer ? (
                        <CheckCircle className="w-4 h-4 mr-2 text-primary" />
                      ) : (
                        <XCircle className="w-4 h-4 mr-2 text-destructive" />
                      )}
                      Your answer: {q.options[answers[index]]}
                    </p>
                    <p className="text-sm text-muted-foreground flex items-center">
                      <CheckCircle className="w-4 h-4 mr-2 text-primary" />
                      Correct answer: {q.options[q.correctAnswer]}
                    </p>
                  </div>
                ))}
              </div>
            </div>
            <PerformanceAnalysis
              score={score}
              totalQuestions={questions.length}
              timeSpent={totalTime}
              questionTimes={questionTimes}
            />
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="container max-w-4xl py-12">
      <Card>
        <CardHeader>
          <CardTitle>{isDemo ? "Demo Quiz: Photosynthesis" : "Quiz"}</CardTitle>
          <Progress value={((currentQuestion + 1) / questions.length) * 100} />
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <h2 className="text-xl font-medium">
              Question {currentQuestion + 1} of {questions.length}
            </h2>
            <p>{questions[currentQuestion].question}</p>
            <div className="grid gap-2">
              {questions[currentQuestion].options.map((option, index) => (
                <Button
                  key={index}
                  variant={selectedAnswer === index ? "default" : "outline"}
                  className="justify-start"
                  onClick={() => handleAnswer(index)}
                >
                  {option}
                </Button>
              ))}
            </div>
          </div>
          <div className="flex justify-between">
            <Button variant="outline" onClick={handlePrevious} disabled={currentQuestion === 0}>
              Previous
            </Button>
            <Button onClick={handleNext} disabled={selectedAnswer === null}>
              {currentQuestion === questions.length - 1 ? "Finish" : "Next"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

