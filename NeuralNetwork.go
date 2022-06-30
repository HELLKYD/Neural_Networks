package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Predictor interface {
	computeInputs() float64
	activate(float64) float64
}

type Neuron struct {
	weights []float64
	inputs  []float64
	bias    float64
}

func (n *Neuron) NewNeuron(inputs []float64) {
	s := rand.NewSource(time.Now().UnixNano())
	ran := rand.New(s)
	n.bias = ran.Float64()
	n.weights = make([]float64, len(inputs))
	for i := 0; i < len(inputs); i++ {
		n.weights[i] = ran.Float64()
	}
	n.inputs = inputs
}

func (n *Neuron) computeInputs() float64 {
	out := float64(0)
	for i := 0; i < len(n.weights); i++ {
		out += n.inputs[i] * n.weights[i]
	}
	out += n.bias
	return out
}

func (n *Neuron) activate(in float64) float64 {
	a := Activator{}
	a.NewActivator(func(in float64) float64 {
		return 1 / (1 + math.Exp(-in))
	}, n)
	return a.activate(in)
}

type Activator struct {
	aF   func(float64) float64
	pred Predictor
}

func (a *Activator) NewActivator(af func(float64) float64, pred Predictor) {
	a.aF = af
	a.pred = pred
}

func (a *Activator) activate(inp float64) float64 {
	return a.aF(inp)
}

type Layer []Neuron

func createLayer(neurons int, input []float64) *Layer {
	var out Layer = make(Layer, 0)
	for i := 0; i < neurons; i++ {
		temp := Neuron{}
		temp.NewNeuron(input)
		out = append(out, temp)
	}
	return &out
}

func (l *Layer) computeInputs() []float64 {
	var out []float64 = make([]float64, 0)
	for _, v := range *l {
		out = append(out, v.computeInputs())
	}
	return out
}

func (l *Layer) activate(in []float64) []float64 {
	out := make([]float64, 0)
	for pos, v := range *l {
		out = append(out, v.activate(in[pos]))
	}
	return out
}

func main() {
	inp := []float64{2.1, 1.2, 4.5}
	var l1 *Layer = createLayer(3, inp)
	out := l1.computeInputs()
	res := l1.activate(out)
	nout := Neuron{}
	nout.NewNeuron(res)
	noutout := nout.computeInputs()
	resout := nout.activate(noutout)
	fmt.Printf("l1: %v\n", l1)
	fmt.Printf("out: %v\n", out)
	fmt.Printf("res: %v\n", res)
	fmt.Printf("resout: %v\n", resout)
}

