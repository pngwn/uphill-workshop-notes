<script lang="ts">
	import { page } from '$app/stores';
	import { onNavigate } from '$app/navigation';

	export let nav_items: { name: string; href: string }[];

	let active = false;
	let el: HTMLSpanElement;

	function click_outside(event: MouseEvent) {
		// Check if the click is outside the element
		if (event.target && !el?.contains(event?.target as Node)) {
			// Your code here: what to do when clicked outside the element
			active = false;
		}
	}

	function esc(event: KeyboardEvent) {
		console.log(event.key);
		if (event.key === 'Escape') {
			active = false;
		}
	}

	onNavigate(() => {
		active = false;
	});
</script>

<svelte:window on:click={click_outside} on:keydown={esc} />

<div bind:this={el} class="menu-wrap">
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<span class="outer-wrap" class:active>
		<span class="inner-wrap" class:active on:click={() => (active = !active)}>
			{nav_items.find((item) => item.href === $page.url.pathname)?.name}
		</span>
	</span>
	{#if active}
		<div class="box-wrap">
			<ul>
				{#each nav_items as { name, href }}
					<li><a {href}>{name}</a></li>
				{/each}
			</ul>
		</div>
	{/if}
</div>

<style>
	.menu-wrap {
		position: relative;
	}

	.box-wrap {
		position: absolute;
		top: 25px;
		left: 0px;
	}

	ul {
		list-style: none;
		width: 200px;
		/* border: 1px solid #eee; */
		/* box-shadow: 0 0 3px 1px rgba(0, 0, 0, 0.02); */
		border-radius: 3px;
		padding: 5px;
		padding-left: 6px;
		/* padding-top: 10px; */
		z-index: 1;
		margin: 1px 0;
		border-top-left-radius: 0;
		margin-top: -1px;
		background-color: #fff;
	}

	ul li {
		margin-bottom: 0.25rem;
	}

	ul li:last-child {
		margin-bottom: 0;
	}

	ul li a {
		text-decoration: none;
		color: #666;
		/* border-bottom: 1px dotted #ccc; */
		text-transform: lowercase;
	}

	ul li a:hover {
		color: blue;

		/* border-bottom: 1px solid black; */
	}

	.inner-wrap {
		cursor: pointer;
		text-transform: lowercase;
		padding-bottom: 0;
		position: relative;
		border-color: transparent;
		text-decoration: none;
		color: black;
		border-bottom: 1px dotted #ccc;
		text-transform: lowercase;
		user-select: none;
	}

	.inner-wrap:hover {
		border-bottom: 1px solid black;
	}

	.inner-wrap.active {
		/* border-color: transparent; */
	}

	.outer-wrap {
		position: relative;

		border-top-left-radius: 3px;
		border-top-right-radius: 3px;
		border: 1px solid transparent;
		border-bottom: none;
		padding: 3px 5px;
		z-index: 2;
		background: #fff;
	}

	.outer-wrap.active {
		/* border-color: transparent; */
		/* border-color: #eee; */
		/* box-shadow: 0 -3px 3px 1px rgba(0, 0, 0, 0.02); */
	}
</style>
