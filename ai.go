package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time" // Added for date filtering

	"github.com/gin-gonic/gin"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type AskRequest struct {
	Question string `json:"question" binding:"required"`
}

var (
	ctx       = context.Background()
	modelID   = os.Getenv("MODEL_ID")
	apiKey    = os.Getenv("GOOGLE_API_KEY")
	chunksDir = "./chunker/chunks"
)

func main() {
	if apiKey == "" || modelID == "" {
		log.Fatal("Environment variables GOOGLE_API_KEY and MODEL_ID must be set.")
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	model := client.GenerativeModel(modelID)

	// --- CHANGE 1: Filter knowledge to only the last 3 days and max 500k characters ---
	// This prevents the "Token Quota Exceeded" error.
	knowledgeBase := loadKnowledgeBase(chunksDir, 3, 500000)

	if knowledgeBase == "" {
		fmt.Println("Warning: No recent knowledge found.")
	} else {
		fmt.Printf("Knowledge Base Loaded: %d characters.\n", len(knowledgeBase))
	}

	model.SystemInstruction = &genai.Content{
		Parts: []genai.Part{
			genai.Text(fmt.Sprintf(`You are a professional Enterprise Assistant. 
			Your primary source of truth is the [LOCAL KNOWLEDGE BASE].
			[LOCAL KNOWLEDGE BASE]:
			%s`, knowledgeBase)),
		},
	}

	r := gin.Default()
	r.POST("/ask", func(c *gin.Context) {
		var req AskRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": "Invalid request body"})
			return
		}

		session := model.StartChat()
		resp, err := session.SendMessage(ctx, genai.Text(req.Question))
		if err != nil {
			// --- CHANGE 2: Improved Error Reporting ---
			c.JSON(500, gin.H{"error": fmt.Sprintf("Gemini Error: %v", err)})
			return
		}

		answer := fmt.Sprintf("%v", resp.Candidates[0].Content.Parts[0])
		c.JSON(200, gin.H{"question": req.Question, "answer": answer, "status": "success"})
	})

	r.Run(":8080")
}

// loadKnowledgeBase now limits by Age and Size
func loadKnowledgeBase(dirPath string, daysBack int, maxChars int) string {
	var builder strings.Builder
	files, err := os.ReadDir(dirPath)
	if err != nil {
		return ""
	}

	cutoff := time.Now().AddDate(0, 0, -daysBack)

	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".txt" {
			info, err := file.Info()
			if err != nil || info.ModTime().Before(cutoff) {
				continue // Skip old files
			}

			path := filepath.Join(dirPath, file.Name())
			content, err := os.ReadFile(path)
			if err == nil {
				builder.WriteString(fmt.Sprintf("\n--- Source: %s ---\n", file.Name()))
				builder.WriteString(string(content))
				builder.WriteString("\n")
			}

			// Stop adding files if we've reached our character limit
			if builder.Len() > maxChars {
				fmt.Println("Truncating knowledge base: Max character limit reached.")
				return builder.String()[:maxChars]
			}
		}
	}

	return builder.String()
}

