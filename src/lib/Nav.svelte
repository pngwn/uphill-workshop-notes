<script>
	import Menu from './Menu.svelte';
	export let section = '';
	export let nav_items;
</script>

<ul>
	<li><Menu {nav_items} /></li>

	<!-- <li><a href="">Section</a></li> -->
	<!-- <li><a href="">Sub Section</a></li> -->
</ul>

<style>
	ul {
		list-style: none;
		display: flex;
		justify-content: flex-start;
		align-items: center;
		margin-left: -5px;
	}

	ul li::after {
		content: '>';
		margin: 0 8px;
		color: #aaa;
		font-size: 12px;
	}

	ul li:last-child::after {
		content: '';
	}

	ul li a {
		text-decoration: none;
		color: black;
		border-bottom: 1px dotted #ccc;
		text-transform: lowercase;
	}

	ul li a:hover {
		border-bottom: 1px solid black;
	}

	li:first-child {
		display: flex;
		align-items: center;
	}
</style>
