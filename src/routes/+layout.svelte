<script>
	import '../normalize.css';
	import '../skeleton.css';
	import Nav from '$lib/Nav.svelte';

	export let nav_items = [
		{ name: 'Open Source', href: '/open-source' },
		{ name: 'SWE Engagament', href: '/swe-engagement' },
		{ name: 'Understanding AI', href: '/understanding-ai' },
		{ name: 'Finding Models', href: '/finding-models' },
		{ name: 'Gradio', href: '/gradio' },
		{ name: 'Using Models', href: '/using-models' },
		{ name: 'HF Inference API', href: '/hf-inference-api' },
		{ name: 'Llama CPP', href: '/llama-cpp' },
		{ name: 'Controlling LLMs', href: '/controlling-llms' },
		{ name: 'Prompt Engineering', href: '/prompt-engineering' },
		{ name: 'Guidance', href: '/guidance' }
	];
</script>

<main>
	<h1>Uphill Notes</h1>
	<hr />

	<Nav {nav_items} />

	<slot></slot>
</main>

<style>
	main {
		margin: 0 auto;
		max-width: 700px;
		padding: 20px;
	}

	h1 {
		font-weight: 600;
		text-transform: uppercase;
		font-size: 2rem;
		letter-spacing: 0.1rem;
		margin-bottom: 0;
	}

	hr {
		margin: 10px 0 5px 0;
	}
</style>
