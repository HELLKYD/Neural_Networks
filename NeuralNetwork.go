package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Neuron struct {
	weights []float64
	bias    float64
	result  float64
}

func (n *Neuron) generateNeuron(numInputs int) {
	s1 := rand.NewSource(time.Now().UnixNano())
	rn := rand.New(s1)
	n.weights = make([]float64, numInputs)
	for i := 0; i < numInputs; i++ {
		n.weights[i] = rn.Float64()
	}
	n.bias = rand.Float64()
}

func (n *Neuron) computeInputs(inputs []float64) float64 {
	output := float64(0)
	go func() {
		for pos, value := range n.weights {
			output += inputs[pos] * value
		}
		output += n.bias
		outValues <- output
	}()
	output = <-outValues
	return output
}

func (n *Neuron) activate(out float64) {
	n.result = sigmoid(out)
}

type Layer struct {
	neurons []Predictor
	input   []float64
	output  []float64
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

type Predictor interface {
	computeInputs(inputs []float64) float64
	activate(out float64)
}

var outValues chan float64 = make(chan float64)

func main() {
	inputs := []float64{200000, 14}

	n1 := Neuron{}
	n1.generateNeuron(len(inputs))

	n2 := Neuron{}
	n2.generateNeuron(len(inputs))

	n3 := Neuron{}
	n3.generateNeuron(len(inputs))

	out1 := n1.computeInputs(inputs)
	out2 := n2.computeInputs(inputs)
	out3 := n3.computeInputs(inputs)

	n1.activate(out1)
	n2.activate(out2)
	n3.activate(out3)

	inputs2 := []float64{n1.result, n2.result, n3.result}

	n21 := Neuron{}
	n21.generateNeuron(len(inputs2))

	n22 := Neuron{}
	n22.generateNeuron(len(inputs2))

	n23 := Neuron{}
	n23.generateNeuron(len(inputs2))

	out21 := n21.computeInputs(inputs2)
	out22 := n22.computeInputs(inputs2)
	out23 := n23.computeInputs(inputs2)

	n21.activate(out21)
	n22.activate(out22)
	n23.activate(out23)

	fmt.Printf("inputs: %v\n", inputs)

	fmt.Printf("n1: %v\n", n1)
	fmt.Printf("n2: %v\n", n2)
	fmt.Printf("n3: %v\n", n3)

	fmt.Printf("out1: %v\n", out1)
	fmt.Printf("out2: %v\n", out2)
	fmt.Printf("out3: %v\n", out3)

	fmt.Printf("Results Layer 1:\n")
	fmt.Printf("result1: %v\n", n1.result)
	fmt.Printf("result2: %v\n", n2.result)
	fmt.Printf("result3: %v\n", n3.result)

	fmt.Printf("n21.result: %v\n", n21.result)
	fmt.Printf("n22.result: %v\n", n22.result)
	fmt.Printf("n23.result: %v\n", n23.result*100000) //normalize function
}
