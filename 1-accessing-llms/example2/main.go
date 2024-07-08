package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/predictionguard/go-client"
)

func main() {
	if err := run(); err != nil {
		log.Fatalln(err)
	}
}

func run() error {
	host := "https://api.predictionguard.com"
	apiKey := os.Getenv("PGKEY")

	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		log.Println(s)
	}

	cln := client.New(logger, host, apiKey)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	prompt := `<|im_start|>system
You are a helpful code assistant.<|im_end|>
<|im_start|>user
Write a Go program that prints out random numbers.<|im_end|>
<|im_start|>assistant<|im_end|>
`

	input := client.CompletionInput{
		Model:       client.Models.Hermes2ProLlama38B,
		Prompt:      prompt,
		MaxTokens:   1000,
		Temperature: 0.3,
	}

	resp, err := cln.Completions(ctx, input)
	if err != nil {
		return fmt.Errorf("ERROR: %w", err)
	}

	fmt.Println(resp.Choices[0].Text)

	return nil
}
