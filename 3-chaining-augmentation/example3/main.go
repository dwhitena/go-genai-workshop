package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/predictionguard/go-client"
)

var host = "https://api.predictionguard.com"
var apiKey = os.Getenv("PGKEY")

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

	// Embed a question for the RAG answer.
	message := "What do I need in order to respond to reviewers?"
	embedding, err := embed("", message)
	if err != nil {
		log.Fatal(err)
	}

	// Search for the relevant chunk.
	chunk, err := search(chunks, *embedding)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(chunk)
}
