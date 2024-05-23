package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	md "github.com/JohannesKaufmann/html-to-markdown"
	"github.com/cohere-ai/cohere-go"
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

// embed vectorizes a user message/query.
func embed(message string, co *cohere.Client) ([]float64, error) {
	res, err := co.Embed(cohere.EmbedOptions{
		Model: "embed-english-light-v2.0",
		Texts: []string{message},
	})
	if err != nil {
		return nil, err
	}
	return res.Embeddings[0], nil
}

// VectorizedChunk is a struct that holds a vectorized chunk.
type VectorizedChunk struct {
	Chunk  string    `json:"chunk"`
	Vector []float64 `json:"vector"`
}

// VectorizedChunks is a slice of vectorized chunks.
type VectorizedChunks []VectorizedChunk

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
func search(chunks VectorizedChunks, embedding []float64) (string, error) {
	outChunk := ""
	var maxSimilarity float64 = 0.0
	for _, c := range chunks {
		distance, err := cosineSimilarity(c.Vector, embedding)
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

// characterTextSplitter takes in a string and splits the string into
// chunks of a given size (split on whitespace) with an overlap of a
// given size of tokens (split on whitespace).
func characterTextSplitter(text string, splitSize int, overlapSize int) []string {

	// Create a slice to hold the chunks.
	chunks := []string{}

	// Split the text into tokens based on whitespace.
	tokens := strings.Split(text, " ")

	// Loop over the tokens creating chunks of size splitSize with an
	// overlap of overlapSize.
	for i := 0; i < len(tokens); i += splitSize - overlapSize {
		end := i + splitSize - overlapSize
		if end > len(tokens) {
			end = len(tokens)
		}
		chunks = append(chunks, strings.Join(tokens[i:end], " "))
	}
	return chunks
}

// websitechunks loads in a website and splits it into chunks with an
// optional start string and end string.
func websiteChunks(website string, start string, end string) ([]string, error) {

	converter := md.NewConverter("", true, nil)

	// Download the Go contribution guide.
	res, err := http.Get(website)
	if err != nil {
		return nil, err
	}
	content, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		return nil, err
	}
	html := string(content)

	// Convert the html to markdown for convenience.
	markdown, err := converter.ConvertString(html)
	if err != nil {
		return nil, err
	}

	// Split the markdown string on any provided start and end strings.
	if start != "" {
		markdown_remaining := strings.Split(markdown, start)[1:]
		markdown = strings.Join(markdown_remaining, "")
	}
	if end != "" {
		markdown = strings.Split(markdown, end)[0]
	}

	// Split the text into reasonable size chunks with an overlap.
	chunks := characterTextSplitter(markdown, 100, 10)
	return chunks, nil
}

func main() {

	// Get the website from the command line arg.
	website := os.Args[1]

	// Download the Go contribution guide in chunks.
	chunks, err := websiteChunks(website, "", "")
	if err != nil {
		log.Fatal(err)
	}

	// Connect to Cohere.
	apiKey := os.Getenv("COHERE_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "COHERE_API_KEY not specified")
		os.Exit(1)
	}
	co, err := cohere.CreateClient(apiKey)
	if err != nil {
		log.Fatal(err)
	}

	// Embed the website chunks.
	vectorizedChunks := VectorizedChunks{}
	if len(chunks) > 20 {

		// Batch requests to cohere in batches of 20 or less chunks.
		for i := 0; i < len(chunks); i += 20 {
			end := i + 20
			if end > len(chunks) {
				end = len(chunks)
			}
			res, err := co.Embed(cohere.EmbedOptions{
				Model: "embed-english-light-v2.0",
				Texts: chunks[i:end],
			})
			if err != nil {
				log.Fatal(err)
			}

			// Add the vectorized chunk to the vectorized chunks.
			for j, chunk := range chunks[i:end] {
				vectorizedChunks = append(vectorizedChunks, VectorizedChunk{
					Chunk:  chunk,
					Vector: res.Embeddings[j],
				})
			}
		}
	} else {
		res, err := co.Embed(cohere.EmbedOptions{
			Model: "embed-english-light-v2.0",
			Texts: chunks,
		})
		if err != nil {
			log.Fatal(err)
		}
		for j, chunk := range chunks {
			vectorizedChunks = append(vectorizedChunks, VectorizedChunk{
				Chunk:  chunk,
				Vector: res.Embeddings[j],
			})
		}
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
		embedding, err := embed(input, co)
		if err != nil {
			log.Fatal(err)
		}

		// Search for the relevant chunk.
		chunk, err := search(vectorizedChunks, embedding)
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
