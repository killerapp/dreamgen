interface Env {
	DREAM_BUCKET: R2Bucket;
}

export default {
	async fetch(request: Request, env: Env) {
		// Only allow GET requests
		if (request.method !== 'GET') {
			return new Response('Method not allowed', { status: 405 });
		}

		// Fetch the object using your bucket binding
		const object = await env.DREAM_BUCKET.get("ComfyUI_00372_.png");
		if (!object) {
			return new Response('Image not found', { status: 404 });
		}

		// Set up headers
		const headers = new Headers();
		headers.set('Content-Type', 'image/png');
		headers.set('Cache-Control', 'public, max-age=86400'); // Cache for 1 day
		headers.set('Access-Control-Allow-Origin', '*'); // Allow CORS

		return new Response(object.body, { 
			headers,
			status: 200 
		});
	}
}
