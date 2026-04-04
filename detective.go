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
	"golang.org/x/time/rate"
	"google.golang.org/api/iterator"
)

func main() {
	ctx := context.Background()
	projectID := os.Getenv("GOOGLE_CLOUD_PROJECT") // e.g. "ford-prd-123"
	daysToLookBack := 1
	requestsPerSecond := 15.0

	client, err := logadmin.NewClient(ctx, projectID)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Set Timeframe
	lookback := time.Now().AddDate(0, 0, -daysToLookBack).Format(time.RFC3339)

	filter := fmt.Sprintf(
		`resource.type="cloud_run_revision" AND severity>=ERROR AND timestamp >= "%s"`,
		lookback,
	)

	fmt.Printf("🕵️  Scanning last %d days for errors in %s...\n", daysToLookBack, projectID)

	limiter := rate.NewLimiter(rate.Limit(requestsPerSecond), 1)

	it := client.Entries(ctx, logadmin.Filter(filter), logadmin.NewestFirst())

	errorGroups := make(map[string]int)
	firstSeen := make(map[string]string)

	for {

		if err := limiter.Wait(ctx); err != nil {
			log.Fatalf("Rate limiter error: %v", err)
		}

		entry, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			log.Fatalf("Error iterating: %v", err)
		}

		msg := fmt.Sprintf("%v", entry.Payload)
		errorGroups[msg]++
		if _, exists := firstSeen[msg]; !exists {
			firstSeen[msg] = entry.Timestamp.Format("Jan 02 15:04")
		}
	}

	fmt.Println("\n--- SUMMARY FOR AI CONTEXT ---")
	if len(errorGroups) == 0 {
		fmt.Printf("No errors found in the last %d days. System is healthy.\n", daysToLookBack)
		return
	}

	for msg, count := range errorGroups {
		fmt.Printf("[%d occurrences] First seen: %s | Message: %s\n",
			count, firstSeen[msg], msg)
	}
}
