package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
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
	return fmt.Sprintf(`Context: "%s"

Question: "%s"
`, context, question)
}

// VectorizedChunk is a struct that holds a vectorized chunk.
type VectorizedChunk struct {
	Id       int       `json:"id"`
	Chunk    string    `json:"chunk"`
	Vector   []float64 `json:"vector"`
	Metadata string    `json:"metadata"`
}

// VectorizedChunks is a slice of vectorized chunks.
type VectorizedChunks []VectorizedChunk

func embed(imageLink string, text string) (*VectorizedChunk, error) {

	logger := func(ctx context.Context, msg string, v ...any) {
		//s := fmt.Sprintf("msg: %s", msg)
		//log.Println(s)
	}

	cln := client.New(logger, host, apiKey)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var image client.ImageNetwork
	if imageLink != "" {
		imageParsed, err := client.NewImageNetwork(imageLink)
		if err != nil {
			return nil, fmt.Errorf("ERROR: %w", err)
		}
		image = imageParsed
	}

	input := []client.EmbeddingInput{
		{
			Text: text,
		},
	}
	if imageLink != "" {
		input[0].Image = image
	}

	resp, err := cln.Embedding(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("ERROR: %w", err)
	}

	return &VectorizedChunk{
		Chunk:  text,
		Vector: resp.Data[0].Embedding,
	}, nil
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(a []float64, b []float64) (cosine float64, err error) {
	count := 0
	length_a := len(a)
	length_b := len(b)
	if length_a > length_b {
		count = length_a
	} else {
		count = length_b
	}
	sumA := 0.0
	s1 := 0.0
	s2 := 0.0
	for k := 0; k < count; k++ {
		if k >= length_a {
			s2 += math.Pow(b[k], 2)
			continue
		}
		if k >= length_b {
			s1 += math.Pow(a[k], 2)
			continue
		}
		sumA += a[k] * b[k]
		s1 += math.Pow(a[k], 2)
		s2 += math.Pow(b[k], 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0, errors.New("vectors should not be null (all zeros)")
	}
	return sumA / (math.Sqrt(s1) * math.Sqrt(s2)), nil
}

// search through the vectorized chunks to find the most similar chunk.
func search(chunks VectorizedChunks, embedding VectorizedChunk) (string, error) {
	outChunk := ""
	var maxSimilarity float64 = 0.0
	for _, c := range chunks {
		distance, err := cosineSimilarity(c.Vector, embedding.Vector)
		if err != nil {
			return "", err
		}
		if distance > maxSimilarity {
			outChunk = c.Chunk
			maxSimilarity = distance
		}
	}
	return outChunk, nil
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

	input := client.ChatSSEInput{
		Model: client.Models.Hermes2ProLlama38B,
		Messages: []client.ChatInputMessage{
			{
				Role:    client.Roles.System,
				Content: "Read the context provided by the user and answer their question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, respond \"Sorry I had trouble answering this question, based on the information I found\".",
			},
			{
				Role: client.Roles.User,
				Content: qAPromptTemplate(
					string(queryContext),
					query,
				),
			},
		},
		MaxTokens:   1000,
		Temperature: 0.3,
	}

	ch := make(chan client.ChatSSE, 1000)

	err := cln.ChatSSE(ctx, input, ch)
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

	// Open the JSON file and load in the vectorized embeddings.
	f, err := os.Open("../example2/chunks.json")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Load it into the chunks value.
	var chunks VectorizedChunks
	if err := json.NewDecoder(f).Decode(&chunks); err != nil {
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

		// Embed a question for the RAG answer.
		embedding, err := embed("", input)
		if err != nil {
			log.Fatal(err)
		}

		// Search for the relevant chunk.
		chunk, err := search(chunks, *embedding)
		if err != nil {
			log.Fatal(err)
		}

		// Print the bot response.
		fmt.Print("\n🤖: ")
		if err := run(input, string(chunk)); err != nil {
			log.Fatalln(err)
		}
		fmt.Print("\n\n")
	}
}
