import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Timer, BarChart2 } from "lucide-react"

interface PerformanceAnalysisProps {
  score: number
  totalQuestions: number
  timeSpent: number
  questionTimes?: number[]
}

export function PerformanceAnalysis({ score, totalQuestions, timeSpent, questionTimes }: PerformanceAnalysisProps) {
  const percentageScore = (score / totalQuestions) * 100
  const averageTimePerQuestion = timeSpent / totalQuestions

  return (
    <Card className="w-full mt-6">
      <CardHeader>
        <CardTitle>Performance Analysis</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold flex items-center">
            <BarChart2 className="w-5 h-5 mr-2" />
            Score
          </h3>
          <Progress value={percentageScore} className="w-full mt-2" />
          <p className="text-sm text-muted-foreground mt-1">
            You scored {score} out of {totalQuestions} ({percentageScore.toFixed(1)}%)
          </p>
        </div>
        <div>
          <h3 className="text-lg font-semibold flex items-center">
            <Timer className="w-5 h-5 mr-2" />
            Time Spent
          </h3>
          <p className="text-sm text-muted-foreground">Total time: {timeSpent.toFixed(2)} seconds</p>
          <p className="text-sm text-muted-foreground">
            Average time per question: {averageTimePerQuestion.toFixed(2)} seconds
          </p>
        </div>
        {questionTimes && questionTimes.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold flex items-center">
              <Timer className="w-5 h-5 mr-2" />
              Time per Question
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {questionTimes.map((time, index) => (
                <div
                  key={index}
                  className="text-sm text-muted-foreground p-2 rounded-md"
                  style={{
                    backgroundImage: `repeating-linear-gradient(45deg, transparent, transparent ${time}px, rgba(0,0,0,0.05) ${time}px, rgba(0,0,0,0.05) ${time * 2}px)`,
                  }}
                >
                  Question {index + 1}: {time.toFixed(2)}s
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

