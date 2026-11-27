# Web Tool Interaction & Agent Architecture Summary
**Date:** 2026-11-25

## 1. Overview
This document summarizes the technical discussion regarding the development of an AI agent capable of web interaction, specifically for performing complex tasks like booking camping sites on legacy web platforms (e.g., Recreation.gov).

## 2. Initial Tasks & Code Examples
The session began with the creation of a standard sorting algorithm to establish baseline capabilities.

### Bubble Sort Implementation (JavaScript)
The agent successfully implemented and tested a Bubble Sort algorithm.
```javascript
/**
 * Implements the Bubble Sort algorithm to sort an array of numbers in ascending order.
 * @param {number[]} arr - The array of numbers to be sorted.
 * @returns {number[]} - The sorted array.
 */
function bubbleSort(arr) {
    let n = arr.length;
    let swapped;
    let sortedArr = [...arr];

    for (let i = 0; i < n; i++) {
        swapped = false;
        for (let j = 0; j < n - i - 1; j++) {
            if (sortedArr[j] > sortedArr[j + 1]) {
                let temp = sortedArr[j];
                sortedArr[j] = sortedArr[j + 1];
                sortedArr[j + 1] = temp;
                swapped = true;
            }
        }
        if (!swapped) break;
    }
    return sortedArr;
}
```

## 3. Hardware & Infrastructure Analysis
The agent is being deployed on a specific hardware configuration:
- **Controller Box:** Intel 6700T CPU + 16GB DDR4 RAM.
- **Inference Box (LAN):** Ryzen 5900 + RX 6800 (Running Ollama).

**Key Takeaway:** Because the "thinking" (LLM inference) is offloaded to the Ryzen box, the 6700T machine acts as a dedicated controller. It can efficiently handle the "Browser Host" duties (Playwright/Headless Chrome) without significant performance degradation.

## 4. Web Search & Interaction Strategy
We evaluated several ways to provide the agent with internet access:
- **Search Engines:** Tavily is recommended over standard search (DuckDuckGo) because it provides "cleaned" content (Markdown/Text) rather than just raw links, reducing token bloat.
- **Browser Engine:** **Playwright** is preferred over Selenium.
- **Optimization:** To keep the 6700T responsive, the agent must run in **Headless Mode** and use **Request Interception** to block images and ads.

### Optimized Playwright Snippet
```python
from playwright.sync_api import sync_playwright

def run_agent_browser():
    with sync_playwright() as p:
        # Launching headless = zero window management overhead
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Block images and ads to save 6700T CPU cycles
        page.route("**/*.{png,jpg,jpeg,gif,webp,svg}", lambda route: route.abort())
        page.route("**/google-analytics*", lambda route: route.abort())
        page.route("**/doubleclick*", lambda route: route.abort())

        page.goto("https://example-booking-site.com")
        # ... logic ...
        browser.close()
```

## 5. Agent Architecture for Complex Bookings
To handle multi-step tasks (like Recreation.gov bookings), the recommended architecture is:

1.  **Observe-Reason-Act (ReAct) Loop:** The agent takes a snapshot of the page (Markdown + Accessibility Tree), reasons about the goal, and decides on the next action.
2.  **Macro-Tools:** Instead of the LLM controlling every mouse click, the Python harness provides high-level "Macro-Tools" (e.g., `select_date()`, `add_to_cart()`).
3.  **State Management:** The agent must detect "Success" or "Error" messages in each loop to decide if it needs to pivot (e.g., if a site is already booked).
4.  **Security:** Sensitive data (Credit Cards) should be pulled from a local vault by a secure tool rather than being handled directly by the LLM prompt.
