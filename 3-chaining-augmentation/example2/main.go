package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	md "github.com/JohannesKaufmann/html-to-markdown"
	cohere "github.com/cohere-ai/cohere-go"
)

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
	Chunk  string    `json:"chunk"`
	Vector []float64 `json:"vector"`
}

// VectorizedChunks is a slice of vectorized chunks.
type VectorizedChunks []VectorizedChunk

func main() {

	// Download the Go contribution guide in chunks.
	chunks, err := websiteChunks("https://go.dev/doc/contribute", "# Contribution Guide", "")
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
