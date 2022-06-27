import java.util.Random;

public class NeuralNetwork {

	public static void main(String[] args) {
		float[] inp = { 2.1f, 1.2f, 4.5f };
		Neuron n1 = new Neuron(inp);
		float out = n1.computeInputs();
		System.out.println(out);
		float res = n1.activate();
		System.out.println(res);
	}

	private static interface Predictor {
		float computeInputs();

		float activate();

		float getInput();
	}

	private static class Neuron implements Predictor {
		float[] weights;
		float[] inputs;
		float bias;
		float out;

		public Neuron(float[] inputs) {
			this.bias = new Random().nextFloat();
			this.weights = new float[inputs.length];
			for (int i = 0; i < inputs.length; i++) {
				weights[i] = new Random().nextFloat();
			}
			this.inputs = inputs;
		}

		public float computeInputs() {
			float out = 0;
			for (int i = 0; i < this.inputs.length; i++) {
				out += this.inputs[i] * this.weights[i];
			}
			out += this.bias;
			this.out = out;
			return out;
		}

		public float activate() {
			return new Activator<Neuron>((in) -> (float) (1 / (1 + Math.exp(-in))), this).activate();
		}

		public float getInput() {
			return this.out;
		}
	}

	@FunctionalInterface
	private static interface ActivationFunction {
		float activate(float in);
	}

	private static class Activator<T extends Predictor> {
		ActivationFunction aF;
		T predictor;

		public Activator(ActivationFunction af, T pred) {
			this.aF = af;
			this.predictor = pred;
		}

		public float activate() {
			return this.aF.activate(this.predictor.getInput());
		}
	}

}

