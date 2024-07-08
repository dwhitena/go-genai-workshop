package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	md "github.com/JohannesKaufmann/html-to-markdown"
	"github.com/predictionguard/go-client"
)

var host = "https://api.predictionguard.com"
var apiKey = os.Getenv("PGKEY")

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
		s := fmt.Sprintf("msg: %s", msg)
		log.Println(s)
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

func main() {

	// Download the Go contribution guide in chunks.
	chunks, err := websiteChunks("https://go.dev/doc/contribute", "# Contribution Guide", "")
	if err != nil {
		log.Fatal(err)
	}

	// Embed the website chunks.
	vectorizedChunks := VectorizedChunks{}
	for i, chunk := range chunks {
		fmt.Printf("Embedding chunk %d of %d\n", i+1, len(chunks))
		vectorizedChunk, err := embed("", chunk)
		if err != nil {
			log.Fatal(err)
		}
		vectorizedChunks = append(vectorizedChunks, *vectorizedChunk)
		vectorizedChunks[i].Id = i
		vectorizedChunks[i].Metadata = chunk
	}

	// Marshal the chunks.
	outJSON, err := json.MarshalIndent(vectorizedChunks, "", "  ")
	if err != nil {
		log.Fatal(err)
	}

	// Output the marshalled data to a JSON file.
	if err := os.WriteFile("chunks.json", outJSON, 0644); err != nil {
		log.Fatal(err)
	}
}
