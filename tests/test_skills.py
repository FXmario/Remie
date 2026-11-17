"""Skills core tests also runnable with the standard-library unittest runner."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from remie import skills
from remie.tools.common import tool_working_directory
from remie.tools.skills import activate_skill_tool


class SkillsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.home = self.base / "home"
        self.project = self.base / "project"
        self.home.mkdir()
        self.project.mkdir()
        (self.project / ".git").mkdir()
        for patcher in (patch.object(Path, "home", return_value=self.home),
                        patch.object(skills, "trust_file", return_value=self.base / "config" / "trust.json")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_skill(self, base=None, name="review", client=".agents", body="Review carefully.", description="Review changes"):
        path = (base or self.project) / client / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")
        return path

    def test_project_metadata_hidden_until_approved(self):
        self.make_skill()
        self.assertEqual(skills.discover(self.project)[0], {})
        self.assertEqual(skills.catalog(self.project), "")
        self.assertIn("error", skills.activate("review", self.project))
        skills.trust_project(self.project)
        self.assertIn("review", skills.discover(self.project)[0])

    def test_user_skills_available_without_project_trust(self):
        self.make_skill(self.home)
        self.assertIn("review", skills.discover(self.project)[0])
        self.assertFalse(skills.is_trusted(self.project))

    def test_trust_is_project_specific_and_survives_reload(self):
        skills.trust_project(self.project)
        skills.trust_project(self.project)
        self.assertEqual(len(json.loads(skills.trust_file().read_text())), 1)
        child = self.project / "src"
        child.mkdir()
        self.assertTrue(skills.is_trusted(child))
        self.assertFalse(skills.is_trusted(self.home))

    def test_project_overrides_user_and_native_wins_within_scope(self):
        self.make_skill(self.home, body="user")
        self.make_skill(body="shared")
        native = self.make_skill(client=".remie", body="native")
        skills.trust_project(self.project)
        found, diagnostics = skills.discover(self.project)
        self.assertEqual(found["review"].location, native)
        self.assertTrue(any("Shadowed" in d for d in diagnostics))
        self.assertEqual(skills.activate("review", self.project)["instructions"], "native")

    def test_catalog_does_not_include_instructions(self):
        self.make_skill(self.home, body="SECRET_BODY")
        catalog = skills.catalog(self.project)
        self.assertIn("Review changes", catalog)
        self.assertNotIn("SECRET_BODY", catalog)

    def test_invalid_yaml_and_missing_description_reported(self):
        path = self.make_skill(self.home)
        for text in ("---\nname: review\n---\ntext", "---\nname: [\n---\ntext", "no frontmatter"):
            path.write_text(text)
            found, diagnostics = skills.discover(self.project)
            self.assertFalse(found)
            self.assertTrue(diagnostics)

    def test_name_mismatch_warns_but_loads(self):
        path = self.make_skill(self.home)
        path.write_text(path.read_text().replace("name: review", "name: other"))
        found, diagnostics = skills.discover(self.project)
        self.assertIn("other", found)
        self.assertTrue(diagnostics)

    def test_activation_lists_resources_without_reading_or_executing(self):
        path = self.make_skill(self.home)
        script = path.parent / "scripts" / "danger.sh"
        script.parent.mkdir()
        script.write_text("exit 123")
        result = skills.activate("review", self.project)
        self.assertEqual(result["resources"], ["scripts/danger.sh"])
        self.assertNotIn("exit 123", json.dumps(result))
        self.assertEqual(result["directory"], str(path.parent))

    def test_activation_reads_latest_file(self):
        path = self.make_skill(self.home)
        skills.discover(self.project)
        path.write_text(path.read_text().replace("Review carefully.", "Updated instructions"))
        self.assertEqual(skills.activate("review", self.project)["instructions"], "Updated instructions")

    def test_symlinked_skill_and_resource_are_ignored(self):
        path = self.make_skill(self.home)
        link = path.parent.parent / "linked"
        link.symlink_to(path.parent, target_is_directory=True)
        (path.parent / "external").symlink_to(self.project, target_is_directory=True)
        found, _ = skills.discover(self.project)
        self.assertEqual(len(found), 1)
        self.assertEqual(skills.activate("review", self.project)["resources"], [])

    def test_large_files_rejected(self):
        path = self.make_skill(self.home, body="x" * skills.MAX_FILE_BYTES)
        self.assertFalse(skills.discover(self.project)[0])
        self.assertRaises(ValueError, skills.parse_skill, path)

    def test_tool_uses_tab_directory(self):
        self.make_skill()
        token = tool_working_directory.set(self.project)
        self.addCleanup(tool_working_directory.reset, token)
        self.assertIn("error", activate_skill_tool("review"))
        skills.trust_project(self.project)
        self.assertTrue(activate_skill_tool("review")["skill_content"])

    def test_compaction_preserves_native_text_and_explicit_activations(self):
        data = {"skill_content": True, "name": "review", "instructions": "Keep me"}
        native = {"role": "tool", "name": "activate_skill", "content": json.dumps(data)}
        text = {"role": "user", "content": f"tool_result({json.dumps(data)})"}
        for message in (native, text, skills.skill_message(data)):
            preserved = skills.preserved_messages([message], [])
            self.assertEqual(skills.active_contents(preserved), {"review": data})
            self.assertEqual(skills.preserved_messages(preserved, [message]), [])
            self.assertEqual(len(skills.preserved_messages(preserved + preserved, [])), 1)

    def test_unknown_skill_is_error(self):
        self.assertIn("error", skills.activate("missing", self.project))

    def test_prompt_catalog_uses_explicit_tab_directory(self):
        from remie.prompts import build_system_prompt
        self.make_skill(body="DO_NOT_EAGERLY_LOAD")
        skills.trust_project(self.project)
        prompt = build_system_prompt(native_tools=True, tab_context={"working_directory": str(self.project)})
        self.assertIn("Available skills", prompt)
        self.assertIn("Review changes", prompt)
        self.assertNotIn("DO_NOT_EAGERLY_LOAD", prompt)
        other = build_system_prompt(native_tools=True, tab_context={"working_directory": str(self.home)})
        self.assertNotIn("Available skills", other)

    def test_activation_registered_for_both_protocols(self):
        from remie.tools.registry import TOOL_REGISTRY, TOOL_PARAMETERS, get_tool_str_representation
        self.assertIs(TOOL_REGISTRY["activate_skill"], activate_skill_tool)
        self.assertEqual(TOOL_PARAMETERS["activate_skill"]["required"], ["name"])
        self.assertIn("activate_skill", get_tool_str_representation("activate_skill"))

    def test_slash_registration(self):
        from remie.tui.slash_commands import resolve_slash_command, slash_command_matches
        self.assertEqual(resolve_slash_command("/skills").name, "skills")
        self.assertIn("skills", [c.name for c in slash_command_matches("/ski")])
