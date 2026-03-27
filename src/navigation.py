NAV_ITEMS = {
    "🏠  Overview": "Data Upload",
    "🗄️  Data": "Data Exploration",
    "⚙️  AutoML": "Training",
    "🧪  Experiments": "Experiments",
    "📦  Registry & Deploy": "Prediction",
    "📈  Monitoring": "History (MLflow)",
}


def init_navigation_state(session_state: dict, nav_items: dict | None = None) -> None:
    """Initialize persistent menu state to avoid rerun/radio race conditions."""
    items = nav_items or NAV_ITEMS

    if "menu_page" not in session_state:
        session_state["menu_page"] = "Data Upload"

    if "menu_label" not in session_state:
        session_state["menu_label"] = next(
            (k for k, v in items.items() if v == session_state.get("menu_page")),
            "🏠  Overview",
        )


def sync_navigation_selection(session_state: dict, selected_nav_label: str, nav_items: dict | None = None) -> str:
    """Persist selected navigation label/page and return the selected page key."""
    items = nav_items or NAV_ITEMS
    menu = items[selected_nav_label]
    session_state["menu_page"] = menu
    session_state["menu_label"] = selected_nav_label
    return menu
