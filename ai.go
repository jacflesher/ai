package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// --- 1. Configuration ---

type AskRequest struct {
	Question string `json:"question" binding:"required"`
}

var (
	ctx     = context.Background()
	modelID = os.Getenv("MODEL_ID")
	apiKey  = os.Getenv("GOOGLE_API_KEY")
	// This points to the folder where your Python chunker saves .txt files
	chunksDir = "./chunker/chunks"
)

func main() {
	if apiKey == "" || modelID == "" {
		log.Fatal("Environment variables GOOGLE_API_KEY and MODEL_ID must be set.")
	}

	// 2. Initialize Gemini Client
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	model := client.GenerativeModel(modelID)

	// 3. The "Knowledge Loader"
	// This function reads all your processed chunks and prepares the context.
	knowledgeBase := loadKnowledgeBase(chunksDir)
	if knowledgeBase == "" {
		fmt.Println("Warning: No knowledge found in", chunksDir)
	} else {
		fmt.Printf("Knowledge Base Loaded: %d characters of local context.\n", len(knowledgeBase))
	}

	// 4. Set the System Instruction (The "Source of Truth")
	model.SystemInstruction = &genai.Content{
		Parts: []genai.Part{
			genai.Text(fmt.Sprintf(`You are a professional Enterprise Assistant. 
			Your primary source of truth is the [LOCAL KNOWLEDGE BASE] provided below.
			
			RULES:
			1. If the [LOCAL KNOWLEDGE BASE] contradicts your training, follow the knowledge base.
			2. If the answer is not in the knowledge base, use your general knowledge but mention it's not in the local docs.
			3. Keep answers concise and technical.

			[LOCAL KNOWLEDGE BASE]:
			%s`, knowledgeBase)),
		},
	}

	// 5. Setup the API Server
	r := gin.Default()

	r.POST("/ask", func(c *gin.Context) {
		var req AskRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": "Invalid request body"})
			return
		}

		// Use a Chat Session to handle conversation flow
		session := model.StartChat()
		resp, err := session.SendMessage(ctx, genai.Text(req.Question))
		if err != nil {
			c.JSON(500, gin.H{"error": fmt.Sprintf("Gemini Error: %v", err)})
			return
		}

		// Extract response text
		answer := fmt.Sprintf("%v", resp.Candidates[0].Content.Parts[0])

		c.JSON(200, gin.H{
			"question": req.Question,
			"answer":   answer,
			"status":   "success",
		})
	})

	fmt.Println("Enterprise Modular Server running on :8080...")
	r.Run(":8080")
}

// loadKnowledgeBase scans the directory for all .txt files and merges them
func loadKnowledgeBase(dirPath string) string {
	var builder strings.Builder

	files, err := os.ReadDir(dirPath)
	if err != nil {
		return ""
	}

	for _, file := range files {
		// Only process text chunks
		if !file.IsDir() && filepath.Ext(file.Name()) == ".txt" {
			path := filepath.Join(dirPath, file.Name())
			content, err := os.ReadFile(path)
			if err == nil {
				builder.WriteString(fmt.Sprintf("\n--- Source: %s ---\n", file.Name()))
				builder.WriteString(string(content))
				builder.WriteString("\n")
			}
		}
	}

	return builder.String()
}
