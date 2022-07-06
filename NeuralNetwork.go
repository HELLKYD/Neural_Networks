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

type Network struct {
	inputLayer     Layer
	hiddenLayers   []Layer
	outputLayer    Layer
	tempLayerOut   []float64
	inputs         []float64
	neuronCountOut float64
	result         []float64
}

func (n *Network) useLayers(layerCount int, neuronCount int) {
	n.inputLayer = *createLayer(neuronCount, n.inputs)
	inLayerOut := n.inputLayer.computeInputs()
	inLayerRes := n.inputLayer.activate(inLayerOut)
	n.tempLayerOut = inLayerRes
	for i := 0; i < layerCount; i++ {
		temp_layer := *createLayer(neuronCount, n.tempLayerOut)
		n.hiddenLayers = append(n.hiddenLayers, temp_layer)
		temp_out := temp_layer.computeInputs()
		n.tempLayerOut = temp_layer.activate(temp_out)
	}
	fl := *createLayer(int(n.neuronCountOut), n.tempLayerOut)
	n.outputLayer = fl
	fl_out := fl.computeInputs()
	fl_res := fl.activate(fl_out)
	n.result = append(n.result, fl_res...)
}

func newNetwork(inp []float64, neuronsPerLayer int, neuronCountOut int, layerCount int) *Network {
	out := Network{inputs: inp, neuronCountOut: float64(neuronCountOut),
		tempLayerOut: make([]float64, 0),
		hiddenLayers: make([]Layer, 0),
		result:       make([]float64, 0),
	}
	out.useLayers(layerCount, neuronsPerLayer)
	return &out
}

func main() {
	inp := []float64{2.1, 1.2, 4.5}
	net := *newNetwork(inp, 3, 1, 3)
	fmt.Printf("result: %v\n", net.result)
}
