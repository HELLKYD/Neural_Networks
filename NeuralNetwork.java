import java.util.Random;

public class NeuralNetwork {
    public static void main(String[] args) {
        float[] inp = {2.1f, 1.2f, 4.5f};
        Layer l1 = new Layer(3, inp);
        float[] out = l1.computeInputs();
        float[] res = l1.activate(out);

        Neuron fNeuron = new Neuron(res);
        float fOut = fNeuron.computeInputs();
        float fRes = fNeuron.activate(fOut);
        System.out.println(fRes);
    }

    static interface Predictor {
        float computeInputs();
        float activate(float x);
    }

    static class Neuron implements Predictor {
        float[] weights;
        float[] inputs;
        float bias;

        public Neuron(float[] inputs) {
            Random r = new Random();
            this.bias = r.nextFloat();
            this.weights = new float[inputs.length];
            for(int i = 0; i < inputs.length; i++) {
                this.weights[i] = r.nextFloat();
            }
            this.inputs = inputs;
        }

        public float computeInputs() {
            float out = 0.0f;
            for(int i = 0; i < this.weights.length; i++) {
                out += this.inputs[i] * this.weights[i];
            }
            out += this.bias;
            return out;
        }

        public float activate(float in) {
            return new Activator((x) -> (float) (1 / (1 + Math.exp(-x))), this).activate(in);
        }
    }

    @FunctionalInterface
    static interface ActivationFunction {
        float function(float x);
    }

    static class Activator {
        ActivationFunction aF;
        Predictor pred;

        public Activator(ActivationFunction af, Predictor pred) {
            this.aF = af;
            this.pred = pred;
        }

        public float activate(float in) {
            return this.aF.function(in);
        }
    }

    static class Layer {
        Neuron[] neurons;

        public Layer(int numNeurons, float[] inputs) {
            neurons = new Neuron[numNeurons];
            for (int i = 0; i < numNeurons; i++) {
                Neuron temp = new Neuron(inputs);
                neurons[i] = temp;
            }
        }

        public float[] computeInputs() {
            float out[] = new float[neurons.length];
            for(int i = 0; i < out.length; i++) {
                out[i] = neurons[i].computeInputs();
            }
            return out;
        }

        public float[] activate(float[] in) {
            float out[] = new float[this.neurons.length];
            for(int i = 0; i < out.length; i++) {
                out[i] = neurons[i].activate(in[i]);
            }
            return out;
        }
    }
}
