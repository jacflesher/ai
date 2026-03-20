// go get cloud.google.com/go/logging
// go run detective.go > ./chunker/chunks/report.txt
// ./ask.sh "based on report.txt, what issues am i having in my project?"

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"cloud.google.com/go/logging/logadmin"
	"google.golang.org/api/iterator"
)

func main() {
	ctx := context.Background()
	projectID := os.Getenv("GOOGLE_CLOUD_PROJECT") // e.g. "ford-prd-123"

	// 1. Initialize the Admin Client
	// This uses your existing 'gcloud' auth on your laptop
	client, err := logadmin.NewClient(ctx, projectID)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// 2. Set the "Cold Case" Timeframe (7 Days)
	lookback := time.Now().AddDate(0, 0, -7).Format(time.RFC3339)

	// 3. The Filter: Only Errors in Cloud Run
	// This is the same language you use in the Logs Explorer UI
	filter := fmt.Sprintf(
		`resource.type="cloud_run_revision" AND severity>=ERROR AND timestamp >= "%s"`,
		lookback,
	)

	fmt.Printf("🕵️  Scanning last 7 days for errors in %s...\n", projectID)

	// 4. Fetch the Logs
	it := client.Entries(ctx, logadmin.Filter(filter), logadmin.NewestFirst())

	// 5. The "Anti-Goldfish" Logic: Frequency Counting
	// We use a Map (like a HashMap in Java) to count unique errors
	errorGroups := make(map[string]int)
	firstSeen := make(map[string]string)

	for {
		entry, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			log.Fatalf("Error iterating: %v", err)
		}

		// Convert payload to string (handles both Text and JSON logs)
		msg := fmt.Sprintf("%v", entry.Payload)
		
		// Group them!
		errorGroups[msg]++
		if _, exists := firstSeen[msg]; !exists {
			firstSeen[msg] = entry.Timestamp.Format("Jan 02 15:04")
		}
	}

	// 6. Output the "Case File" for Gemini
	fmt.Println("\n--- SUMMARY FOR AI CONTEXT ---")
	if len(errorGroups) == 0 {
		fmt.Println("No errors found in the last 7 days. System is healthy.")
		return
	}

	for msg, count := range errorGroups {
		fmt.Printf("[%d occurrences] First seen: %s | Message: %s\n", 
			count, firstSeen[msg], msg)
	}
}
