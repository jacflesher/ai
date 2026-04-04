package main

// go get cloud.google.com/go/vertexai/genai
//
// gcloud auth application-default login
//
// export GOOGLE_CLOUD_PROJECT="ford-afd20ec4c8a9b3a7599c2ef8"
// export LOCATION="us-central1"
// export MODEL_ID="gemini-1.5-flash"

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"cloud.google.com/go/vertexai/genai" // Vertex AI SDK
	"github.com/gin-gonic/gin"
)

type AskRequest struct {
	Question string `json:"question" binding:"required"`
}

var (
	ctx       = context.Background()
	projectID = os.Getenv("GOOGLE_CLOUD_PROJECT") // e.g. "my-company-project"
	location  = os.Getenv("LOCATION")             // e.g. "us-central1"
	modelID   = os.Getenv("MODEL_ID")             // e.g. "gemini-1.5-flash"
	chunksDir = "./chunker/chunks"
)

func main() {
	// Validation for Enterprise environment variables
	if projectID == "" || location == "" || modelID == "" {
		log.Fatal("Environment variables GOOGLE_CLOUD_PROJECT, LOCATION, and MODEL_ID must be set.")
	}

	// Vertex AI client uses Project ID and Location instead of an API Key
	client, err := genai.NewClient(ctx, projectID, location)
	if err != nil {
		log.Fatalf("Failed to create Vertex AI client: %v", err)
	}
	defer client.Close()

	model := client.GenerativeModel(modelID)

	// Load knowledge base (same logic as before)
	knowledgeBase := loadKnowledgeBase(chunksDir, 3, 500000)

	if knowledgeBase == "" {
		fmt.Println("Warning: No recent knowledge found.")
	} else {
		fmt.Printf("Knowledge Base Loaded: %d characters.\n", len(knowledgeBase))
	}

	// Set System Instructions for the model
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

		// Start a chat session
		session := model.StartChat()
		resp, err := session.SendMessage(ctx, genai.Text(req.Question))
		if err != nil {
			c.JSON(500, gin.H{"error": fmt.Sprintf("Vertex AI Error: %v", err)})
			return
		}

		if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
			c.JSON(500, gin.H{"error": "No response parts returned from model"})
			return
		}

		answer := fmt.Sprintf("%v", resp.Candidates[0].Content.Parts[0])
		c.JSON(200, gin.H{"question": req.Question, "answer": answer, "status": "success"})
	})

	fmt.Println("🚀 Enterprise AI Server starting on :8080...")
	r.Run(":8080")
}

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
				continue
			}

			path := filepath.Join(dirPath, file.Name())
			content, err := os.ReadFile(path)
			if err == nil {
				builder.WriteString(fmt.Sprintf("\n--- Source: %s ---\n", file.Name()))
				builder.WriteString(string(content))
				builder.WriteString("\n")
			}

			if builder.Len() > maxChars {
				fmt.Println("Truncating knowledge base: Max character limit reached.")
				return builder.String()[:maxChars]
			}
		}
	}
	return builder.String()
}
