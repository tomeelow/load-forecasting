# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright==1.65.2"]
# ///
"""Open the hosted dashboard in a real browser so Streamlit Community Cloud keeps it up.

A plain HTTP GET is not enough: the host only serves a static shell, and a sleeping app
stays asleep until someone presses the wake button on it. So this loads the page the way
a visitor would, presses that button if it is there, and waits until the app itself has
rendered — which fails loudly if the app is broken rather than merely asleep.

Run by `.github/workflows/keep-awake.yml`; `uv run deploy/keep_awake.py <url>` locally.
"""

import logging
import sys
import time

from playwright.sync_api import Page, sync_playwright

WAKE_BUTTON = "Yes, get this app back up!"
APP_CONTAINER = '[data-testid="stAppViewContainer"]'
# A cold start reinstalls the locked dependencies before the page renders.
RENDER_TIMEOUT_S = 600
POLL_S = 5

log = logging.getLogger("keep_awake")


def app_rendered(page: Page) -> bool:
    """The app is served in an iframe under the host's shell, so look in every frame."""
    return any(frame.locator(APP_CONTAINER).count() for frame in page.frames)


def keep_awake(url: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")

        deadline = time.monotonic() + RENDER_TIMEOUT_S
        woke = False
        while not app_rendered(page):
            button = page.get_by_role("button", name=WAKE_BUTTON)
            if not woke and button.count():
                log.info("App was asleep — pressing the wake button")
                button.click()
                woke = True
            if time.monotonic() > deadline:
                raise TimeoutError(f"{url} did not render within {RENDER_TIMEOUT_S}s")
            page.wait_for_timeout(POLL_S * 1000)

        log.info("App is up (%s)", "woken from sleep" if woke else "was already awake")
        browser.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    keep_awake(sys.argv[1])
