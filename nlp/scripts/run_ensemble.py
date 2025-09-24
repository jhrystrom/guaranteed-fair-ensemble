import argparse
import itertools
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Run bertc.py for combinations of languages and fairness metrics"
    )
    parser.add_argument(
        "--fair",
        nargs="+",
        default=["equal_opportunity"],
        help="Fairness metrics (default: equal_opportunity)",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["pl"],
        choices=["pl"],
        help="Languages",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="gender",
        choices=["country", "gender"],
    )

    args = parser.parse_args()

    for fair, lang in itertools.product(args.fair, args.langs):
        cmd = [
            "uv",
            "run",
            "bertc.py",
            f"oofair={args.group},lang={lang},ensemble,fair={fair}",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
