console.debug("DevTools Scraper frontend bootstrap");

// Test function for browser profiling - generates a ~200ms long task
function generateLongTask() {
    console.time('longTask');
    const primes = [];
    for (let i = 2; i < 50000; i++) {
        let isPrime = true;
        for (let j = 2; j <= Math.sqrt(i); j++) {
            if (i % j === 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) primes.push(i);
    }
    console.timeEnd('longTask');
    console.log(`Found ${primes.length} primes`);
    return primes.length;
}

// Expose for testing from browser console
window.generateLongTask = generateLongTask;
