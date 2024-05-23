package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/predictionguard/go-client"
)

// Define the API details to access the LLM.
var host = "https://api.predictionguard.com"
var apiKey = os.Getenv("PGKEY")

// qAPromptTemplate is a template for a question and answer prompt.
func qAPromptTemplate(context, question string) string {
	return fmt.Sprintf(`Read the context below and answer the question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, respond "Sorry I had trouble answering this question, based on the information I found."

Context: "%s"

Question: "%s"
`, context, question)
}

func run(query, queryContext string) error {

	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		//log.Println(s)
	}

	cln := client.New(logger, host, apiKey)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	input := []client.ChatMessage{
		{
			Role: client.Roles.User,
			Content: qAPromptTemplate(
				string(queryContext),
				query,
			),
		},
	}

	ch := make(chan client.ChatSSE, 1000)

	err := cln.ChatSSE(ctx, client.Models.Hermes2ProMistral7B, input, 1000, 0.1, ch)
	if err != nil {
		return fmt.Errorf("ERROR: %w", err)
	}

	for resp := range ch {
		for _, choice := range resp.Choices {
			fmt.Print(choice.Delta.Content)
		}
	}

	return nil
}

func main() {

	// Get the context file from a command line argument.
	if len(os.Args) < 2 {
		log.Fatal("Please provide a context file as an argument.")
	}
	contextFile := os.Args[1]

	// Read in the context from a file.
	context, err := os.ReadFile(contextFile)
	if err != nil {
		log.Fatal(err)
	}

	// Start a cycle of listening for questions and responding to the questions.
	fmt.Println("")
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("🧑: ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()

		// Exit if we type "exit".
		if strings.ToLower(input) == "exit" {
			break
		}

		// Print the bot response.
		fmt.Print("\n🤖: ")
		if err := run(input, string(context)); err != nil {
			log.Fatalln(err)
		}
		fmt.Print("\n\n")
	}
}
