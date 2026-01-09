#!/usr/bin/env python3
import logging
from typing import Optional

from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from tau2.gym.gym_agent import AgentGymEnv, UserGymEnv
from tau2.run import get_options, load_task_splits, load_tasks
from tau2.utils.tools import is_functional_tool_call, parse_functional_tool_call

# Initialize Rich console
console = Console()


def disable_logging():
    """Disable all logging during manual mode for cleaner CLI output."""
    # Disable loguru logging
    logger.remove()
    # Add a handler that does nothing (suppresses all logs)
    logger.add(lambda msg: None, level="CRITICAL")
    
    # Disable standard Python logging
    logging.getLogger().setLevel(logging.CRITICAL)
    # Disable all loggers to be extra sure
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).disabled = True


def enable_logging():
    """Re-enable logging after manual mode."""
    # Re-enable loguru logging
    logger.remove()
    # Re-add default console handler
    logger.add(lambda msg: print(msg), level="INFO")
    
    # Re-enable standard Python logging
    logging.getLogger().setLevel(logging.INFO)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.INFO)
        logging.getLogger(name).disabled = False


def display_domains():
    """Display available domains and let user choose one."""
    options = get_options()
    domains = options.domains

    # Create a table for domains
    table = Table(title="🎯 Available Domains", box=box.ROUNDED)
    table.add_column("Number", style="cyan", justify="center")
    table.add_column("Domain", style="green", justify="left")

    for i, domain in enumerate(domains, 1):
        table.add_row(str(i), domain)

    console.print(table)

    while True:
        try:
            choice = Prompt.ask(
                f"\n[bold blue]Select a domain[/bold blue] (1-{len(domains)})",
                default="1",
            )
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(domains):
                return domains[choice_idx]
            else:
                console.print(
                    f"[red]Please enter a number between 1 and {len(domains)}[/red]"
                )
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def display_task_split_set(domain: str) -> Optional[str]:
    """Display available task split sets for the domain and let user choose one."""
    task_splits = load_task_splits(domain)
    if task_splits is None:
        console.print(
            f"[red]No task splits found for domain '{domain}', using full task set[/red]"
        )
        return None

    # Create a table for task splits
    table = Table(title="🔧 Available Task Splits", box=box.ROUNDED)
    table.add_column("Number", style="cyan", justify="center")
    table.add_column("Task Split", style="green", justify="left")

    for i, task_split in enumerate(task_splits, 1):
        table.add_row(str(i), task_split)

    console.print(table)

    while True:
        try:
            choice = Prompt.ask(
                f"\n[bold blue]Select a task split set[/bold blue] (1-{len(task_splits)})",
                default="1",
            )
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(task_splits):
                return list(task_splits.keys())[choice_idx]
            else:
                console.print(
                    f"[red]Please enter a number between 1 and {len(task_splits)}[/red]"
                )
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def display_tasks(domain: str, task_split_set: Optional[str] = None):
    """Display available tasks for the domain and let user choose one."""
    # Try to load tasks for the domain
    try:
        tasks = load_tasks(domain, task_split_set)
    except Exception as e:
        console.print(f"[red]Error loading tasks for domain '{domain}': {e}[/red]")
        # Try alternative task sets
        options = get_options()
        task_sets = [ts for ts in options.task_sets if domain in ts]
        if task_sets:
            console.print(
                f"[yellow]Available task sets for {domain}: {task_sets}[/yellow]"
            )
            task_set = task_sets[0]  # Use first available task set
            console.print(f"[green]Using task set: {task_set}[/green]")
            tasks = load_tasks(task_set, task_split_set)
        else:
            raise ValueError(f"No task sets found for domain '{domain}'")

    # Create a table for tasks
    table = Table(title=f"📋 Available Tasks for {domain}", box=box.ROUNDED)
    table.add_column("Number", style="cyan", justify="center")
    table.add_column("Task ID", style="green", justify="left")
    table.add_column("Description", style="white", justify="left")

    for i, task in enumerate(tasks, 1):
        # Safely handle task description
        try:
            if hasattr(task, "description") and task.description:
                if isinstance(task.description, str):
                    description = task.description
                else:
                    description = str(task.description)
            else:
                description = "No description available"
        except Exception:
            description = "Description unavailable"

        table.add_row(str(i), task.id, description)

    console.print(table)

    while True:
        try:
            choice = Prompt.ask(
                f"\n[bold blue]Select a task[/bold blue] (1-{len(tasks)})", default="1"
            )
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(tasks):
                return tasks[choice_idx]
            else:
                console.print(
                    f"[red]Please enter a number between 1 and {len(tasks)}[/red]"
                )
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def display_policy(policy: str):
    """Display the agent policy to the user."""
    if not policy:
        console.print(Panel("No policy available for this domain.", style="red"))
        return

    # Create a panel for the policy
    policy_panel = Panel(
        policy,
        title="📋 Agent Policy",
        border_style="yellow",
        box=box.ROUNDED,
        width=100,
    )
    console.print(policy_panel)


def display_tools(tools):
    """Display available tools to the user."""
    if not tools:
        console.print(Panel("No tools available for this domain.", style="red"))
        return

    # Create a table for tools
    table = Table(title="🔧 Available Tools", box=box.ROUNDED)
    table.add_column("Tool Name", style="cyan", justify="left")
    table.add_column("Description", style="white", justify="left")
    table.add_column("Parameters", style="yellow", justify="left")

    for tool in tools:
        # Get the description from the tool
        if hasattr(tool, "short_desc") and tool.short_desc:
            desc = tool.short_desc
        elif hasattr(tool, "long_desc") and tool.long_desc:
            desc = tool.long_desc
        else:
            desc = "No description available"

        # Show parameters if available
        params_text = ""
        if hasattr(tool, "params") and tool.params:
            try:
                params_schema = tool.params.model_json_schema()
                if "properties" in params_schema:
                    param_names = list(params_schema["properties"].keys())
                    if param_names:
                        params_text = ", ".join(param_names)
            except Exception:
                pass

        table.add_row(tool.name, desc, params_text)

    console.print(table)


def format_observation(observation: str, step_count: int):
    """Format the observation for better display."""
    if not observation.strip():
        console.print(Panel("No observation available", style="red"))
        return

    # Create a panel for the observation
    title = f"STEP {step_count} - CURRENT OBSERVATION"

    # Split by lines and format each message
    formatted_lines = []
    lines = observation.strip().split("\n")

    for line in lines:
        if line.strip():
            if line.startswith("user:"):
                formatted_lines.append(
                    f"[bold blue]👤 USER:[/bold blue] {line[5:].strip()}"
                )
            elif line.startswith("assistant:"):
                formatted_lines.append(
                    f"[bold green]🤖 ASSISTANT:[/bold green] {line[10:].strip()}"
                )
            elif line.startswith("system:"):
                formatted_lines.append(
                    f"[bold yellow]⚙️  SYSTEM:[/bold yellow] {line[7:].strip()}"
                )
            else:
                formatted_lines.append(f"[white]📝 {line.strip()}[/white]")

    content = "\n\n".join(formatted_lines)
    panel = Panel(content, title=title, border_style="blue", box=box.ROUNDED)
    console.print(panel)


def get_user_action(
    env, step_count: int, tools, policy: str, task=None, solo_mode=False, play_as_user=False
) -> str:
    """Get the next action from the user."""
    role_text = "user" if play_as_user else "agent"
    console.print(
        f"\n[bold cyan] STEP {step_count} - Enter your action as the {role_text}:[/bold cyan]"
    )
    help_text = "[dim](Type 'quit' to exit, 'help' for commands, 'tools' to see available tools"
    if play_as_user:
        help_text += ", 'scenario' to see your goal, or add '###STOP###' to end"
    else:
        help_text += ", 'policy' to see agent policy"
    if solo_mode:
        help_text += ", 'ticket' to see task ticket"
    help_text += ")[/dim]"
    console.print(help_text)

    while True:
        action = Prompt.ask("[bold green]Action[/bold green]")
        if action.lower() == "quit":
            return None
        elif action.lower() == "help":
            help_content = """[bold]📋 Available commands:[/bold]
• Type any text to send as your response
• 'quit': Exit the simulation
• 'help': Show this help message
• 'tools': Show available tools"""
            if play_as_user:
                help_content += "\n• 'scenario': Show your task goal"
            else:
                help_content += "\n• 'policy': Show agent policy"
            if solo_mode:
                help_content += "\n• 'ticket': Show task ticket"

            help_content += "\n\n[bold]💡 Tips:[/bold]"
            if play_as_user:
                help_content += "\n• You are the customer - make requests and ask questions"
                help_content += "\n• Be natural and conversational"
                help_content += "\n• The automated agent will respond to you"
                help_content += "\n• Type 'scenario' to remind yourself of your goal"
                help_content += "\n• You can also use user tools (if available) by typing their names and parameters"
                help_content += "\n• Example: [cyan]check_balance()[/cyan]"
                help_content += "\n\n[bold]🛑 To end the conversation:[/bold]"
                help_content += "\n• Add [cyan]###STOP###[/cyan] to signal you're satisfied"
                help_content += "\n• Example: [cyan]Thanks, that's perfect! ###STOP###[/cyan]"
                help_content += "\n• Or add [cyan]###TRANSFER###[/cyan] to request a human agent"
                help_content += "\n• Or add [cyan]###OUT-OF-SCOPE###[/cyan] if request is out of scope"
            else:
                help_content += "\n• You can use tools by typing their names and parameters"
                help_content += "\n• Example: [cyan]search_flights(origin=\"NYC\", destination=\"LAX\")[/cyan]"
                if solo_mode:
                    help_content += "\n• In solo mode, work through the ticket step by step"
                else:
                    help_content += "\n• Be conversational and helpful to the user"
                help_content += "\n• Follow the agent policy guidelines"

            help_panel = Panel(
                help_content,
                title="🆘 Help",
                border_style="green",
                box=box.ROUNDED,
            )
            console.print(help_panel)
        elif action.lower() == "tools":
            display_tools(tools)
        elif action.lower() == "policy":
            display_policy(policy)
        elif action.lower() == "ticket":
            if solo_mode and task:
                display_ticket(task)
            else:
                console.print(
                    "[yellow]Ticket command is only available in solo mode.[/yellow]"
                )
        elif action.lower() == "scenario":
            if play_as_user and task:
                display_user_scenario(task)
            else:
                console.print(
                    "[yellow]Scenario command is only available when playing as user.[/yellow]"
                )
        elif action:
            # Check if the action looks like a functional tool call
            if is_functional_tool_call(action):
                try:
                    # Parse the functional tool call
                    tool_call = parse_functional_tool_call(action)
                    console.print(
                        f"[green]🔧 Parsed tool call:[/green] [cyan]{tool_call.name}[/cyan] "
                        f"with arguments: [yellow]{tool_call.arguments}[/yellow]"
                    )
                    # Return the action as-is for now - the environment will handle the tool call
                    return action
                except (ValueError, SyntaxError) as e:
                    console.print(f"[red]❌ Error parsing tool call: {e}[/red]")
                    console.print(
                        "[yellow]Please check the format. Example: function_name(arg1='value1', arg2=123)[/yellow]"
                    )
                    continue
            return action
        else:
            console.print("[red]Please enter a valid action[/red]")


def display_role_selection():
    """Display role selection (agent or user) and let user choose."""
    role_panel = Panel(
        """[bold]🎭 Choose your role:[/bold]

[bold blue]Play as Agent:[/bold blue] You are the AI assistant
• Respond to user requests
• Use tools to help solve problems
• Follow the agent policy
• Options: Normal mode (with user simulator) or Solo mode (work independently)

[bold green]Play as User:[/bold green] You are the customer
• Make requests and ask questions
• An automated LLM agent will respond to you
• Experience the conversation from the user's perspective
• Great for testing agent behavior
• Note: Solo mode is NOT available when playing as user""",
        title="🎯 Role Selection",
        border_style="cyan",
        box=box.ROUNDED,
    )
    console.print(role_panel)

    return Confirm.ask("\n[bold blue]Play as User?[/bold blue]", default=False)


def display_mode_selection():
    """Display mode selection (solo or normal) and let user choose.
    
    Note: This is only available when playing as the agent.
    """
    mode_panel = Panel(
        """[bold]🎭 Choose your interaction mode:[/bold]

[bold blue]Normal Mode:[/bold blue] You interact with a simulated user
• The user simulator will respond based on the task scenario
• You'll have conversations back and forth

[bold green]Solo Mode:[/bold green] You work independently on a ticket
• No user interaction - you solve the task directly
• You'll see a ticket with the problem description
• Work through the solution step by step""",
        title="🔧 Mode Selection",
        border_style="cyan",
        box=box.ROUNDED,
    )
    console.print(mode_panel)

    return Confirm.ask("\n[bold blue]Enable Solo Mode?[/bold blue]", default=False)


def get_user_llm_config():
    """Get user LLM configuration (when playing as agent)."""
    llm_panel = Panel(
        """[bold]🤖 User Simulator LLM Configuration:[/bold]

Configure which LLM to use for the user simulator.
Leave empty to use the default LLM.

[dim]Examples: gpt-4, claude-3-sonnet, etc.[/dim]""",
        title="⚙️ LLM Configuration",
        border_style="yellow",
        box=box.ROUNDED,
    )
    console.print(llm_panel)

    user_llm = Prompt.ask(
        "\n[bold blue]Enter User LLM name[/bold blue] (or press Enter for default)",
        default="",
    )

    return user_llm if user_llm.strip() else None


def get_agent_llm_config():
    """Get agent LLM configuration (when playing as user)."""
    llm_panel = Panel(
        """[bold]🤖 Agent LLM Configuration:[/bold]

Configure which LLM to use for the automated agent.
Leave empty to use the default LLM.

[dim]Examples: gpt-4o, claude-3-sonnet, gpt-4, etc.[/dim]""",
        title="⚙️ LLM Configuration",
        border_style="yellow",
        box=box.ROUNDED,
    )
    console.print(llm_panel)

    agent_llm = Prompt.ask(
        "\n[bold blue]Enter Agent LLM name[/bold blue] (or press Enter for default)",
        default="",
    )

    return agent_llm if agent_llm.strip() else None


def display_ticket(task):
    """Display the task ticket when in solo mode."""
    if not hasattr(task, "ticket") or not task.ticket:
        console.print(Panel("No ticket available for this task.", style="yellow"))
        return

    ticket_panel = Panel(
        task.ticket,
        title="🎫 Task Ticket",
        border_style="green",
        box=box.ROUNDED,
        width=100,
    )
    console.print(ticket_panel)


def display_user_scenario(task):
    """Display the user scenario/instructions when playing as user."""
    if not hasattr(task, "user_scenario") or not task.user_scenario:
        console.print(Panel("No user scenario available for this task.", style="yellow"))
        return

    # Get the user scenario content
    if hasattr(task.user_scenario, "instructions"):
        # It's a UserScenario object with persona and instructions
        content = f"[bold]👤 Persona:[/bold]\n{task.user_scenario.persona}\n\n"
        content += f"[bold]📋 Your Goal:[/bold]\n{task.user_scenario.instructions}"
    else:
        # It's a string
        content = str(task.user_scenario)

    scenario_panel = Panel(
        content,
        title="🎯 Your Task (User Scenario)",
        border_style="cyan",
        box=box.ROUNDED,
        width=100,
    )
    console.print(scenario_panel)


def main():
    """Main function for the manual mode."""
    # Disable logging for cleaner CLI output
    disable_logging()

    # Welcome message with Rich styling
    welcome_text = Text()
    welcome_text.append("🎮 Welcome to ", style="bold blue")
    welcome_text.append("Tau2 Manual Mode", style="bold green")
    welcome_text.append("!", style="bold blue")

    welcome_panel = Panel(
        """You will be playing the role of the agent in a domain.
This allows you to interact with the simulation as if you were the AI agent.

[bold]Ready to start your adventure?[/bold] 🚀""",
        title=welcome_text,
        border_style="blue",
        box=box.DOUBLE,
    )

    console.print(welcome_panel)

    try:
        # Step 1: Choose role (agent or user)
        play_as_user = display_role_selection()
        console.print(
            f"\n[green]✅ Selected role:[/green] [bold]{'User' if play_as_user else 'Agent'}[/bold]"
        )

        # Step 2: Choose domain
        domain = display_domains()
        console.print(f"\n[green]✅ Selected domain:[/green] [bold]{domain}[/bold]")

        # Step 3: Choose a task split set
        task_split_set = display_task_split_set(domain)
        console.print(
            f"\n[green]✅ Selected task split set:[/green] [bold]{task_split_set}[/bold]"
        )

        # Step 4: Choose task
        task = display_tasks(domain, task_split_set)
        console.print(f"\n[green]✅ Selected task:[/green] [bold]{task.id}[/bold]")
        if task.description:
            try:
                if isinstance(task.description, str):
                    description = task.description
                else:
                    description = str(task.description)
                console.print(f"[dim]📝 Task description: {description}[/dim]")
            except Exception:
                console.print(
                    "[dim]📝 Task description: [red]Unable to display[/red][/dim]"
                )

        # Step 5: Choose mode (solo or normal) - only if playing as agent
        solo_mode = False
        if not play_as_user:
            solo_mode = display_mode_selection()
            console.print(
                f"\n[green]✅ Selected mode:[/green] [bold]{'Solo' if solo_mode else 'Normal'}[/bold]"
            )
        else:
            console.print(
                f"\n[green]✅ Mode:[/green] [bold]Normal (Solo mode not available when playing as user)[/bold]"
            )

        # Step 6: Get LLM configuration
        user_llm = None
        agent_llm = None
        if play_as_user:
            # Playing as user - configure the automated agent
            agent_llm = get_agent_llm_config()
            if agent_llm:
                console.print(f"\n[green]✅ Agent LLM:[/green] [bold]{agent_llm}[/bold]")
            else:
                console.print(f"\n[green]✅ Agent LLM:[/green] [bold]Default[/bold]")
        else:
            # Playing as agent - configure the user simulator (only for normal mode)
            if not solo_mode:
                user_llm = get_user_llm_config()
                if user_llm:
                    console.print(f"\n[green]✅ User LLM:[/green] [bold]{user_llm}[/bold]")
                else:
                    console.print(f"\n[green]✅ User LLM:[/green] [bold]Default[/bold]")

        # Step 7: Create appropriate GymEnv instance
        with console.status("[bold green]Initializing environment...", spinner="dots"):
            if play_as_user:
                env = UserGymEnv(
                    domain=domain,
                    task_id=task.id,
                    agent_llm=agent_llm
                )
            else:
                env = AgentGymEnv(
                    domain=domain, task_id=task.id, solo_mode=solo_mode, user_llm=user_llm
                )

        # Step 8: Reset environment and get initial observation
        console.print("\n[bold green]🚀 Starting simulation...[/bold green]")
        observation, info = env.reset()

        # Get tools and policy from info dictionary
        if play_as_user:
            # When playing as user, show user tools and agent tools separately
            agent_tools = info.get("agent_tools", [])
            user_tools = info.get("user_tools", [])
            policy = info.get("policy", "")
            
            console.print("\n[bold blue]🤖 Agent Tools:[/bold blue]")
            display_tools(agent_tools)
            
            if user_tools:
                console.print("\n[bold green]👤 Your Tools (as User):[/bold green]")
                display_tools(user_tools)
            else:
                console.print("\n[dim]👤 No user tools available for this domain[/dim]")
            
            console.print("\n[bold yellow]📋 Agent Policy (for reference):[/bold yellow]")
            display_policy(policy)
        else:
            # When playing as agent, show agent tools and policy
            tools = info.get("tools", [])
            policy = info.get("policy", "")
            display_tools(tools)
            display_policy(policy)
            # Set for compatibility
            user_tools = []

        # Step 9: Display ticket if in solo mode, or user scenario if playing as user
        if solo_mode:
            display_ticket(task)
        elif play_as_user:
            display_user_scenario(task)

        # Main interaction loop
        step_count = 0
        while True:
            step_count += 1

            # Display current observation
            format_observation(observation, step_count)

            # Get user action
            current_tools = user_tools if play_as_user else tools
            action = get_user_action(env, step_count, current_tools, policy, task, solo_mode, play_as_user)
            if action is None:
                console.print("[yellow]👋 Exiting simulation...[/yellow]")
                break

            # Step the environment
            try:
                with console.status("[bold green]Processing action...", spinner="dots"):
                    observation, reward, terminated, truncated, info = env.step(action)

                # Update tools and policy from info (in case they changed)
                if play_as_user:
                    agent_tools = info.get("agent_tools", agent_tools)
                    user_tools = info.get("user_tools", user_tools)
                    current_tools = user_tools
                else:
                    tools = info.get("tools", tools)
                    current_tools = tools
                policy = info.get("policy", policy)

                if terminated:
                    console.print(
                        Panel(
                            f"[bold green]🏆 Simulation Completed![/bold green]\n"
                            f"Final reward: [bold yellow]{reward}[/bold yellow]",
                            title="🏁 Simulation Terminated",
                            border_style="green",
                            box=box.ROUNDED,
                        )
                    )
                    break
                elif truncated:
                    console.print(
                        Panel(
                            "[bold yellow]Simulation was truncated (time limit reached)[/bold yellow]",
                            title="⏰ Simulation Truncated",
                            border_style="yellow",
                            box=box.ROUNDED,
                        )
                    )
                    break

            except Exception as e:
                console.print(f"[red]❌ Error during simulation step: {e}[/red]")
                console.print("[yellow]🔄 Continuing with next step...[/yellow]")
                continue

        console.print(
            Panel(
                "[bold green]🎉 Simulation ended. Thank you for playing![/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    except KeyboardInterrupt:
        console.print("\n\n[red]⏹️  Simulation interrupted by user.[/red]")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]❌ Error: {e}[/bold red]\n"
                "Please check your domain and task selection.",
                title="Error",
                border_style="red",
                box=box.ROUNDED,
            )
        )
    finally:
        # Re-enable logging when exiting
        enable_logging()


if __name__ == "__main__":
    main()
